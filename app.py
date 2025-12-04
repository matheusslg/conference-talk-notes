import streamlit as st
from supabase import create_client
from google import genai
from openai import OpenAI
import anthropic
from PIL import Image
import pillow_heif
import os
import tempfile
import io
import json

# Register HEIF/HEIC support with Pillow
pillow_heif.register_heif_opener()

# Page config
st.set_page_config(
    page_title="Conference Talk Notes",
    layout="wide"
)

# ============== Password Authentication ==============

def check_password():
    """Returns True if user entered correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Conference Talk Notes")
    st.caption("AWS re:Invent 2025")
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    # Autofocus password field
    st.markdown('''
    <script>
        // Autofocus password input
        setTimeout(() => {
            const pwInput = document.querySelector('input[type="password"]');
            if (pwInput) pwInput.focus();
        }, 100);
    </script>
    ''', unsafe_allow_html=True)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")
    return False

# Gate the entire app
if not check_password():
    st.stop()

# Constants
SUPPORTED_AUDIO_FORMATS = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Models by provider
LLM_MODELS = {
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "OpenAI": ["gpt-4o", "gpt-4o-mini"],
    "Anthropic": ["claude-sonnet-4-20250514"],
}

# Flatten for transcription (Gemini only for audio)
AVAILABLE_MODELS = LLM_MODELS["Gemini"]

# All models for text generation
ALL_LLM_MODELS = [model for models in LLM_MODELS.values() for model in models]

# Custom CSS - AWS re:Invent 2025 Dark Theme
st.markdown("""
<style>
    /* Import fonts from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600;700&family=Fira+Mono:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

    /* Hide Streamlit top bar, menu, and decorations */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    .stDeployButton {display: none;}

    /* Hide keyboard shortcut hints */
    [data-testid="InputInstructions"] {display: none;}
    .st-emotion-cache-ue6h4q {display: none;}
    div[data-baseweb="tooltip"] {display: none !important;}

    /* Set primary font family globally */
    html, body, [class*="css"] {
        font-family: "Amazon Ember", "EmberDisplay", "Open Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        font-family: "Amazon Ember", "EmberDisplay", "Open Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Apply font to text elements only */
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp input, .stApp textarea {
        font-family: "Amazon Ember", "EmberDisplay", "Open Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Secondary/mono font for code and captions */
    code, pre, .stCode, [data-testid="stCode"] {
        font-family: "Amazon Ember Mono", "EmberMono", "Fira Mono", "Consolas", monospace !important;
    }

    .stCaption, [data-testid="stCaptionContainer"], small {
        font-family: "Amazon Ember Mono", "EmberMono", "Fira Mono", "Consolas", monospace !important;
    }

    /* Global dark theme overrides */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #ffffff;
    }

    /* Force white text throughout */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #ffffff !important;
    }

    /* Secondary/muted text */
    .stCaption, [data-testid="stCaptionContainer"],
    .stApp small, .element-container small {
        color: #a1a1aa !important;
    }

    /* Headers with gradient accent */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Talk cards */
    .talk-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid #2d2d44;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        min-height: 180px;
        display: flex;
        flex-direction: column;
    }

    .talk-card:hover {
        border-color: #9333ea;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(147, 51, 234, 0.3);
    }

    .talk-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.3;
    }

    .talk-card-speaker {
        color: #a1a1aa;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }

    .talk-card-stats {
        display: flex;
        gap: 16px;
        margin-top: auto;
        padding-top: 12px;
        border-top: 1px solid #2d2d44;
    }

    .talk-card-stat {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #a1a1aa;
        font-size: 0.85rem;
    }

    .talk-card-stat-value {
        color: #ec4899;
        font-weight: 600;
    }

    /* Style native Streamlit containers with border */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 16px;
        transition: all 0.3s ease;
    }

    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #9333ea;
        box-shadow: 0 8px 25px rgba(147, 51, 234, 0.2);
    }

    /* Summary container */
    .summary-container {
        background: rgba(147, 51, 234, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #9333ea;
        border: 1px solid #2d2d44;
    }

    /* Content badges */
    .content-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 5px;
    }
    .badge-audio { background-color: rgba(99, 102, 241, 0.2); color: #818cf8; }
    .badge-ocr { background-color: rgba(236, 72, 153, 0.2); color: #f472b6; }
    .badge-vision { background-color: rgba(147, 51, 234, 0.2); color: #a78bfa; }

    /* Upload container */
    .upload-container {
        background: rgba(147, 51, 234, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #9333ea;
    }

    /* Danger zone */
    .danger-zone {
        background: rgba(239, 68, 68, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ef4444;
    }

    /* Header gradient text effect */
    .gradient-header {
        background: linear-gradient(90deg, #9333ea, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
    }

    /* All buttons with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
        border: none !important;
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px #9333ea !important;
    }

    /* Secondary/outline buttons - darker style */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid #9333ea !important;
        color: #ec4899 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(147, 51, 234, 0.2) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%) !important;
        border: none !important;
        color: white !important;
    }

    /* Form submit button */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%) !important;
        border: none !important;
        color: white !important;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #ec4899;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #a1a1aa;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%);
        color: white;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a1a2e;
        border-radius: 8px;
    }

    /* Divider styling */
    hr {
        border-color: #2d2d44;
    }

    /* Form styling */
    [data-testid="stForm"] {
        background: rgba(26, 26, 46, 0.5);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d2d44;
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        border-color: #2d2d44 !important;
    }

    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Selectbox dropdown */
    [data-baseweb="select"] {
        background-color: #1a1a2e !important;
    }

    [data-baseweb="select"] > div {
        background-color: #1a1a2e !important;
        border-color: #2d2d44 !important;
    }

    [data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* Selectbox input container */
    .stSelectbox > div > div {
        background-color: #1a1a2e !important;
        border-color: #2d2d44 !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: #1a1a2e !important;
        border: 1px solid #2d2d44 !important;
    }

    [data-baseweb="popover"] li {
        background-color: #1a1a2e !important;
    }

    [data-baseweb="popover"] li:hover {
        background-color: #2d2d44 !important;
    }

    /* Menu list */
    [data-baseweb="menu"] {
        background-color: #1a1a2e !important;
    }

    [data-baseweb="menu"] li {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }

    [data-baseweb="menu"] li span,
    [data-baseweb="menu"] li div {
        color: #ffffff !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: rgba(147, 51, 234, 0.3) !important;
    }

    /* Selected option highlight */
    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: rgba(147, 51, 234, 0.5) !important;
    }

    /* Force all dropdown option text white */
    [role="listbox"] li,
    [role="listbox"] li *,
    [role="option"],
    [role="option"] *,
    [data-baseweb="menu"] [role="option"],
    [data-baseweb="menu"] [role="option"] * {
        color: #ffffff !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(147, 51, 234, 0.1);
        border-radius: 10px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1a1a2e !important;
        border: 1px solid #2d2d44;
    }

    /* Checkbox and radio */
    .stCheckbox label, .stRadio label {
        color: #ffffff !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
    }

    /* Blockquotes */
    blockquote {
        border-left-color: #9333ea !important;
        color: #e5e7eb !important;
    }

    /* Code blocks */
    code {
        background-color: #2d2d44 !important;
        color: #f472b6 !important;
    }

    /* Links */
    a {
        color: #ec4899 !important;
    }

    a:hover {
        color: #f472b6 !important;
    }

    /* ============== RESPONSIVE DESIGN ============== */

    /* Tablet breakpoint */
    @media (max-width: 992px) {
        /* Reduce padding on main content */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Smaller headers */
        h1 {
            font-size: 1.75rem !important;
        }

        h2, h3 {
            font-size: 1.25rem !important;
        }

        /* Tabs: smaller padding */
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 0.85rem;
        }

        /* Talk cards: reduce min-height */
        .talk-card {
            min-height: 150px;
            padding: 15px;
        }
    }

    /* Mobile breakpoint */
    @media (max-width: 768px) {
        /* Tighter main content padding */
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
        }

        /* Smaller headers for mobile */
        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        /* Tabs: horizontal scroll with smaller tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 10px;
            font-size: 0.8rem;
            white-space: nowrap;
            flex-shrink: 0;
        }


        /* Reduce top padding on mobile */
        .stMain > .block-container,
        .stMainBlockContainer,
        section.main > div {
            padding-top: 2rem !important;
        }


        /* Talk cards: stack vertically, full width */
        .talk-card {
            min-height: auto;
            padding: 12px;
            margin-bottom: 8px;
        }

        /* Containers with border: less padding */
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 12px;
        }

        /* Summary container: less padding */
        .summary-container {
            padding: 12px;
        }

        /* Upload container: less padding */
        .upload-container {
            padding: 12px;
        }

        /* Danger zone: less padding */
        .danger-zone {
            padding: 10px;
        }

        /* Form: less padding */
        [data-testid="stForm"] {
            padding: 12px;
        }

        /* Buttons: ensure touch-friendly size */
        .stButton > button {
            min-height: 44px;
            font-size: 0.9rem;
        }

        /* Metrics: smaller */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }

        /* Chat messages: less padding */
        [data-testid="stChatMessage"] {
            padding: 8px !important;
        }

        /* Expanders: smaller text */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }

        /* File uploader: less padding */
        [data-testid="stFileUploader"] {
            padding: 8px;
        }
    }

    /* Small mobile breakpoint */
    @media (max-width: 480px) {
        /* Even tighter padding */
        .main .block-container {
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }

        /* Smaller headers */
        h1 {
            font-size: 1.3rem !important;
        }

        h2 {
            font-size: 1.1rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }

        /* Tabs: even smaller */
        .stTabs [data-baseweb="tab"] {
            padding: 6px 8px;
            font-size: 0.75rem;
        }

        /* Metrics: even smaller */
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize clients
@st.cache_resource
def init_clients():
    supabase = create_client(
        os.environ.get("SUPABASE_URL", st.secrets.get("SUPABASE_URL", "")),
        os.environ.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_SERVICE_KEY", ""))
    )
    gemini_client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    )

    # OpenAI client (optional)
    openai_key = os.environ.get("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    openai_client = OpenAI(api_key=openai_key) if openai_key else None

    # Anthropic client (optional)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", st.secrets.get("ANTHROPIC_API_KEY", ""))
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None

    return supabase, gemini_client, openai_client, anthropic_client

supabase, gemini_ai, openai_ai, anthropic_ai = init_clients()

# For backward compatibility
ai = gemini_ai

# ============== Helper Functions ==============

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start + overlap >= len(text):
            break
    return chunks

def generate_embedding(text: str) -> list:
    response = ai.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embeddings[0].values

# ============== LLM Abstraction ==============

def generate_with_llm(prompt: str, model: str) -> str:
    """Generate text using the specified LLM model."""
    if model.startswith("gemini"):
        response = gemini_ai.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    elif model.startswith("gpt"):
        if not openai_ai:
            raise ValueError("OpenAI API key not configured")
        response = openai_ai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    elif model.startswith("claude"):
        if not anthropic_ai:
            raise ValueError("Anthropic API key not configured")
        response = anthropic_ai.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unknown model: {model}")

def get_available_models() -> list:
    """Return list of models that have API keys configured."""
    available = LLM_MODELS["Gemini"].copy()  # Gemini always available
    if openai_ai:
        available.extend(LLM_MODELS["OpenAI"])
    if anthropic_ai:
        available.extend(LLM_MODELS["Anthropic"])
    return available

# ============== AI Content Persistence ==============

def save_ai_content(talk_id: str, content_type: str, content: str, model: str):
    """Save AI-generated content (always creates new entry for history)."""
    supabase.from_("talk_ai_content").insert({
        "talk_id": talk_id,
        "content_type": content_type,
        "content": content,
        "model_used": model
    }).execute()

def get_all_ai_content(talk_id: str, content_type: str) -> list:
    """Get all stored AI content for a content type, ordered by newest first."""
    result = supabase.from_("talk_ai_content").select("id, content, model_used, created_at").eq("talk_id", talk_id).eq("content_type", content_type).order("created_at", desc=True).execute()
    return result.data or []

def delete_ai_content(content_id: str):
    """Delete a specific AI content entry."""
    supabase.from_("talk_ai_content").delete().eq("id", content_id).execute()

def get_ai_content(talk_id: str, content_type: str) -> dict:
    """Get most recent AI content. Returns dict with 'content' and 'model_used' or None."""
    result = supabase.from_("talk_ai_content").select("content, model_used, created_at").eq("talk_id", talk_id).eq("content_type", content_type).order("created_at", desc=True).limit(1).execute()
    return result.data[0] if result.data else None

def get_chat_history(talk_id: str) -> list:
    """Get stored chat history for a talk."""
    result = get_ai_content(talk_id, "chat")
    if result and result.get("content"):
        try:
            return json.loads(result["content"])
        except json.JSONDecodeError:
            return []
    return []

def save_chat_history(talk_id: str, history: list, model: str):
    """Save chat history as JSON."""
    save_ai_content(talk_id, "chat", json.dumps(history), model)

# ============== Talk Management ==============

def create_talk(title: str, speaker: str = None) -> str:
    result = supabase.from_("talks").insert({
        "title": title,
        "speaker": speaker,
    }).execute()
    return result.data[0]["id"] if result.data else None

def get_all_talks() -> list:
    result = supabase.from_("talks").select("*").order("created_at", desc=True).execute()
    talks = result.data or []

    # Enrich with chunk counts
    for talk in talks:
        chunks = supabase.from_("talk_chunks").select("content_type, slide_number").eq("talk_id", talk["id"]).execute()
        talk["audio_count"] = len([c for c in (chunks.data or []) if c["content_type"] == "audio_transcript"])
        talk["slide_count"] = len(set([c.get("slide_number") for c in (chunks.data or []) if c.get("slide_number")]))

    return talks

def get_talk_by_id(talk_id: str) -> dict:
    result = supabase.from_("talks").select("*").eq("id", talk_id).single().execute()
    return result.data

def update_talk(talk_id: str, title: str, speaker: str = None) -> bool:
    supabase.from_("talks").update({
        "title": title,
        "speaker": speaker,
        "updated_at": "now()"
    }).eq("id", talk_id).execute()
    return True

def delete_talk(talk_id: str) -> bool:
    supabase.from_("talks").delete().eq("id", talk_id).execute()
    return True

def get_talk_chunks(talk_id: str) -> list:
    result = supabase.from_("talk_chunks").select("*").eq("talk_id", talk_id).order("created_at").execute()
    return result.data or []

def get_uploaded_files(talk_id: str) -> dict:
    """Get list of uploaded files grouped by type."""
    chunks = supabase.from_("talk_chunks").select("source_file, content_type, created_at").eq("talk_id", talk_id).order("created_at").execute()

    files = {"audio": [], "slides": []}
    seen_audio = set()
    seen_slides = set()

    for c in (chunks.data or []):
        filename = c["source_file"]
        if c["content_type"] == "audio_transcript":
            if filename not in seen_audio:
                seen_audio.add(filename)
                files["audio"].append(filename)
        elif c["content_type"] in ("slide_ocr", "slide_vision"):
            if filename not in seen_slides:
                seen_slides.add(filename)
                files["slides"].append(filename)

    return files

# ============== Audio Ingestion ==============

def ingest_audio(file_path: str, file_name: str, talk_id: str, model: str, progress_callback=None) -> dict:
    results = {"status": "success", "messages": [], "chunks": 0}

    try:
        if progress_callback:
            progress_callback(0.1, "Uploading audio to Gemini...")

        uploaded_file = ai.files.upload(file=file_path)

        if progress_callback:
            progress_callback(0.2, "Processing audio file...")

        file = ai.files.get(name=uploaded_file.name)
        while file.state.name == "PROCESSING":
            import time
            time.sleep(2)
            file = ai.files.get(name=uploaded_file.name)

        if file.state.name == "FAILED":
            results["status"] = "error"
            results["messages"].append("File processing failed")
            return results

        if progress_callback:
            progress_callback(0.4, f"Transcribing with {model}...")

        transcription_result = ai.models.generate_content(
            model=model,
            contents=[
                file,
                "Transcribe this audio file. Output only the transcription text, nothing else."
            ]
        )

        transcript_text = transcription_result.text or ""
        results["messages"].append(f"Transcription complete ({len(transcript_text):,} characters)")

        ai.files.delete(name=file.name)

        if progress_callback:
            progress_callback(0.6, "Chunking transcript...")

        chunks = chunk_text(transcript_text, CHUNK_SIZE, CHUNK_OVERLAP)
        results["messages"].append(f"Split into {len(chunks)} chunks")

        if progress_callback:
            progress_callback(0.7, "Generating embeddings and storing...")

        stored_count = 0
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            supabase.from_("talk_chunks").insert({
                "talk_id": talk_id,
                "content": chunk,
                "content_type": "audio_transcript",
                "source_file": file_name,
                "chunk_index": i,
                "embedding": embedding,
            }).execute()

            stored_count += 1
            if progress_callback and i % 5 == 0:
                progress = 0.7 + (0.25 * (i / len(chunks)))
                progress_callback(progress, f"Storing chunk {i+1}/{len(chunks)}...")

        results["chunks"] = stored_count
        results["messages"].append(f"Stored {stored_count} chunks")

        if progress_callback:
            progress_callback(1.0, "Done!")

    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"Error: {str(e)}")

    return results

# ============== Image Ingestion ==============

def extract_talk_info_from_slide(image: Image.Image) -> dict:
    """Extract talk title and speaker from a title slide image."""
    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image,
            """This is a conference presentation title slide. Extract:
1. The talk title (usually the largest text, may include session code like "SVS301")
2. The speaker name(s)

Output as JSON only, no markdown:
{"title": "extracted title", "speaker": "speaker name or null if not found"}

If you cannot identify a title, use: {"title": null, "speaker": null}"""
        ]
    )
    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except:
        return {"title": None, "speaker": None}

def extract_slide_ocr(image: Image.Image) -> str:
    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image,
            """Extract ALL text visible in this slide image.
Output only the raw text, preserving structure with newlines.
Include headers, bullet points, code snippets, labels, and annotations.
If no text is visible, output 'NO_TEXT_FOUND'."""
        ]
    )
    text = response.text or ""
    return "" if text.strip() == "NO_TEXT_FOUND" else text

def describe_slide_vision(image: Image.Image) -> str:
    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image,
            """Describe the visual elements of this presentation slide:
- Diagrams and their components
- Charts/graphs and what they show
- Architecture diagrams with service names
- Icons and their meaning
- Color coding or visual hierarchy

Focus on technical content. Be concise but complete.
If it's just a text-only slide with no visual elements, output 'TEXT_ONLY_SLIDE'."""
        ]
    )
    text = response.text or ""
    return "" if text.strip() == "TEXT_ONLY_SLIDE" else text

def ingest_images(images: list, talk_id: str, progress_callback=None) -> dict:
    results = {"status": "success", "messages": [], "slides_processed": 0}

    try:
        for idx, (image, filename) in enumerate(images):
            if progress_callback:
                progress_callback(idx / len(images), f"Processing slide {idx + 1}/{len(images)}...")

            # Extract OCR text
            ocr_text = extract_slide_ocr(image)
            if ocr_text:
                chunks = chunk_text(ocr_text, 1500, 0) if len(ocr_text) > 1500 else [ocr_text]
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = generate_embedding(chunk)
                    supabase.from_("talk_chunks").insert({
                        "talk_id": talk_id,
                        "content": chunk,
                        "content_type": "slide_ocr",
                        "source_file": filename,
                        "slide_number": idx + 1,
                        "chunk_index": chunk_idx,
                        "embedding": embedding,
                    }).execute()

            # Get vision description
            vision_desc = describe_slide_vision(image)
            if vision_desc:
                embedding = generate_embedding(vision_desc)
                supabase.from_("talk_chunks").insert({
                    "talk_id": talk_id,
                    "content": vision_desc,
                    "content_type": "slide_vision",
                    "source_file": filename,
                    "slide_number": idx + 1,
                    "chunk_index": 0,
                    "embedding": embedding,
                }).execute()

            results["slides_processed"] += 1

        results["messages"].append(f"Processed {results['slides_processed']} slides")

        if progress_callback:
            progress_callback(1.0, "Done!")

    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"Error: {str(e)}")

    return results

# ============== Search ==============

def search_talk_content(query: str, talk_id: str = None, content_types: list = None, match_count: int = 10) -> list:
    query_vector = generate_embedding(query)

    params = {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": match_count,
    }

    if talk_id:
        params["filter_talk_id"] = talk_id
    if content_types:
        params["filter_content_types"] = content_types

    result = supabase.rpc("search_talk_chunks", params).execute()
    return result.data or []

def generate_search_insights(query: str, results: list) -> str:
    if not results:
        return ""

    audio_context = [r for r in results if r['content_type'] == 'audio_transcript']
    ocr_context = [r for r in results if r['content_type'] == 'slide_ocr']
    vision_context = [r for r in results if r['content_type'] == 'slide_vision']

    context_parts = []
    if audio_context:
        context_parts.append("**From Audio Transcripts:**\n" + "\n\n".join([r['content'] for r in audio_context[:3]]))
    if ocr_context:
        context_parts.append("**From Slide Text:**\n" + "\n\n".join([r['content'] for r in ocr_context[:3]]))
    if vision_context:
        context_parts.append("**From Slide Visuals:**\n" + "\n\n".join([r['content'] for r in vision_context[:3]]))

    context = "\n\n---\n\n".join(context_parts)

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Based on the user's question and the relevant content from conference talk materials below, provide a comprehensive answer.

**User's Question:** {query}

**Content from Talk Materials:**
{context}

**Instructions:**
- Synthesize information from audio transcripts AND slide content
- Reference specific slides when relevant (e.g., "As shown in slide 3...")
- Be technical and specific for AWS re:Invent content
- Use bullet points for clarity

Provide your response in markdown format."""
    )
    return response.text

# ============== Insights ==============

def extract_key_quotes(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    """Extract memorable quotes, statistics, and key statements."""
    talk = get_talk_by_id(talk_id)
    chunks = get_talk_chunks(talk_id)

    if not chunks:
        return "No content available."

    audio_text = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
    slides_text = "\n".join([c['content'] for c in chunks if c['content_type'] == 'slide_ocr'])

    prompt = f"""Extract the most memorable and impactful quotes, statistics, and key statements from this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

**Content:**
{audio_text[:8000]}

{slides_text[:3000]}

**Instructions:**
- Extract 5-10 of the most quotable statements
- Include any specific statistics, numbers, or metrics mentioned
- Focus on insights that would be valuable to share with colleagues
- Format each quote with context

Output format:
## Key Quotes

> "Quote here"
— Context or speaker attribution

## Notable Statistics
- Stat 1
- Stat 2

## Memorable Insights
- Insight 1
- Insight 2{f'''

**Additional user instructions:** {custom_instructions}''' if custom_instructions else ''}"""

    result = generate_with_llm(prompt, model)
    save_ai_content(talk_id, "quotes", result, model)
    return result

def extract_action_items(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    """Extract action items, recommendations, and next steps."""
    talk = get_talk_by_id(talk_id)
    chunks = get_talk_chunks(talk_id)

    if not chunks:
        return "No content available."

    audio_text = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
    slides_text = "\n".join([c['content'] for c in chunks if c['content_type'] == 'slide_ocr'])

    prompt = f"""Extract all action items, recommendations, and next steps from this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

**Content:**
{audio_text[:8000]}

{slides_text[:3000]}

**Instructions:**
- Identify explicit recommendations made by the speaker
- Extract any "you should..." or "consider doing..." statements
- Note any AWS services or tools recommended to try
- Include any best practices mentioned
- Categorize by priority or effort level if possible

Output format:
## Immediate Actions
- [ ] Action 1
- [ ] Action 2

## Things to Explore
- [ ] Tool/service to try
- [ ] Concept to learn more about

## Best Practices to Adopt
- [ ] Practice 1
- [ ] Practice 2

## Resources Mentioned
- Link or resource 1
- Link or resource 2{f'''

**Additional user instructions:** {custom_instructions}''' if custom_instructions else ''}"""

    result = generate_with_llm(prompt, model)
    save_ai_content(talk_id, "actions", result, model)
    return result

def chat_with_talk(talk_id: str, question: str, chat_history: list, model: str = "gemini-2.5-flash") -> str:
    """Answer questions about this specific talk using RAG."""
    # Search for relevant context
    results = search_talk_content(question, talk_id=talk_id, match_count=5)

    if not results:
        return "I couldn't find relevant information in this talk to answer your question. Try rephrasing or ask something else about the talk content."

    context = "\n\n".join([
        f"[{r['content_type']}] {r['content']}"
        for r in results
    ])

    # Build chat history context
    history_text = ""
    if chat_history:
        history_text = "\n\n**Previous conversation:**\n"
        for msg in chat_history[-6:]:  # Last 3 exchanges
            history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

    prompt = f"""You are a helpful assistant answering questions about a specific AWS re:Invent talk.

**Relevant content from the talk:**
{context}
{history_text}
**Current question:** {question}

**Instructions:**
- Answer based ONLY on the provided talk content
- Be specific and reference the talk material
- If the answer isn't in the content, say so
- Keep responses concise but helpful
- Use markdown formatting"""

    return generate_with_llm(prompt, model)

# ============== Summary & Export ==============

def generate_talk_summary(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    talk = get_talk_by_id(talk_id)
    chunks = get_talk_chunks(talk_id)

    if not chunks:
        return "No content available to summarize."

    audio_text = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
    slides_text = "\n\n".join([
        f"Slide {c['slide_number']}: {c['content']}"
        for c in chunks if c['content_type'] in ('slide_ocr', 'slide_vision')
    ])

    prompt = f"""Create a comprehensive summary of this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

**Audio Transcript (excerpt):**
{audio_text[:8000]}

**Slide Content:**
{slides_text[:4000]}

**Generate the following sections:**

## Summary
Brief 2-3 paragraph overview of the talk

## Key Topics
- Bullet points of main topics covered

## AWS Services Mentioned
- List any AWS services discussed

## Key Takeaways
- Main learnings and insights

## Recommended Actions
- What to do with this knowledge

Be technical and specific. Use markdown formatting.{f'''

**Additional user instructions:** {custom_instructions}''' if custom_instructions else ''}"""

    result = generate_with_llm(prompt, model)

    # Save to database
    save_ai_content(talk_id, "summary", result, model)

    return result

def export_to_markdown(talk_id: str, include_transcript: bool = True) -> str:
    talk = get_talk_by_id(talk_id)
    chunks = get_talk_chunks(talk_id)
    summary = generate_talk_summary(talk_id)

    md = f"""# {talk['title']}

**Speaker:** {talk.get('speaker', 'Unknown')}
**Event:** {talk.get('event', 'AWS re:Invent 2025')}

---

{summary}

---

## Slide Content

"""

    # Group by slide number
    slides = {}
    for c in chunks:
        if c.get('slide_number'):
            num = c['slide_number']
            if num not in slides:
                slides[num] = {'ocr': '', 'vision': ''}
            if c['content_type'] == 'slide_ocr':
                slides[num]['ocr'] += c['content'] + "\n"
            elif c['content_type'] == 'slide_vision':
                slides[num]['vision'] = c['content']

    for num in sorted(slides.keys()):
        md += f"""### Slide {num}

**Text:**
{slides[num]['ocr'].strip() or '_No text extracted_'}

**Visual Description:**
{slides[num]['vision'] or '_No visual elements_'}

---

"""

    if include_transcript:
        transcript = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
        if transcript:
            md += f"""## Full Transcript

{transcript}
"""

    return md

# ============== Session State ==============

if "active_view" not in st.session_state:
    st.session_state.active_view = "talks"
if "selected_talk" not in st.session_state:
    st.session_state.selected_talk = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.5-flash"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get talks
talks = get_all_talks()

# Get available models (based on configured API keys)
available_models = get_available_models()

# ============== Top Navigation ==============

st.markdown("#### Conference Talk Notes")
st.caption("AWS re:Invent 2025")
nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("Talks", key="nav_talks", use_container_width=True, icon=":material/home:"):
        st.session_state.active_view = "talks"
        st.session_state.selected_talk = None
        st.rerun()
with nav_cols[1]:
    if st.button("Search", key="nav_search", use_container_width=True, icon=":material/search:"):
        st.session_state.active_view = "search"
        st.session_state.selected_talk = None
        st.rerun()
with nav_cols[2]:
    st.session_state.selected_model = st.selectbox(
        "Model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        key="nav_model",
        label_visibility="collapsed"
    )

# ============== Main Content ==============

if st.session_state.active_view == "talks":

    # New Talk form in main content
    st.markdown("### Add New Talk")

    create_method = st.radio("Create method", ["Manual", "From Title Slide"], horizontal=True, label_visibility="collapsed")

    if create_method == "Manual":
        with st.form("new_talk_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                new_title = st.text_input("Talk Title", placeholder="SVS301 - Building Serverless Apps...")
            with col2:
                new_speaker = st.text_input("Speaker (optional)", placeholder="John Doe")

            if st.form_submit_button("Create Talk", type="primary", use_container_width=True, icon=":material/add:"):
                if new_title:
                    talk_id = create_talk(new_title, new_speaker or None)
                    if talk_id:
                        st.session_state.selected_talk = talk_id
                        st.session_state.active_view = "talk_detail"
                        st.rerun()
                else:
                    st.warning("Please enter a talk title")
    else:
        # Create from title slide
        title_slide = st.file_uploader(
            "Upload title slide",
            type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
            key="title_slide_uploader"
        )

        if title_slide:
            img = Image.open(io.BytesIO(title_slide.getvalue()))
            st.image(img, width=300)

            if st.button("Extract Info", icon=":material/auto_awesome:", use_container_width=True):
                with st.spinner("Analyzing slide..."):
                    info = extract_talk_info_from_slide(img)
                    st.session_state.extracted_title = info.get("title") or ""
                    st.session_state.extracted_speaker = info.get("speaker") or ""
                st.rerun()

        if "extracted_title" in st.session_state:
            with st.form("create_from_slide_form"):
                extracted_title = st.text_input("Title", value=st.session_state.extracted_title)
                extracted_speaker = st.text_input("Speaker", value=st.session_state.extracted_speaker)

                if st.form_submit_button("Create Talk", type="primary", use_container_width=True, icon=":material/add:"):
                    if extracted_title:
                        talk_id = create_talk(extracted_title, extracted_speaker or None)
                        del st.session_state.extracted_title
                        del st.session_state.extracted_speaker
                        st.session_state.selected_talk = talk_id
                        st.session_state.active_view = "talk_detail"
                        st.rerun()
                    else:
                        st.warning("Please enter a talk title")

    st.divider()

    if talks:
        st.markdown("### Your Talks")

        # Create grid layout - 3 columns
        cols = st.columns(3)
        for idx, t in enumerate(talks):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{t['title']}**")
                    if t.get('speaker'):
                        st.caption(f"Speaker: {t['speaker']}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Audio", t.get('audio_count', 0))
                    with col_b:
                        st.metric("Slides", t.get('slide_count', 0))

                    if st.button("Open", key=f"open_{t['id']}", use_container_width=True, icon=":material/arrow_forward:"):
                        st.session_state.selected_talk = t["id"]
                        st.session_state.active_view = "talk_detail"
                        st.rerun()

elif st.session_state.active_view == "search":
    st.title("Search All Talks")
    st.markdown("Find topics across all your conference notes")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search query", placeholder="What were the key points about Lambda cold starts?", label_visibility="collapsed")
    with col2:
        num_results = st.selectbox("Results", [5, 10, 15, 20], label_visibility="collapsed")

    # Content type filter
    content_filter = st.multiselect(
        "Filter by content type",
        ["audio_transcript", "slide_ocr", "slide_vision"],
        default=["audio_transcript", "slide_ocr", "slide_vision"],
        format_func=lambda x: {"audio_transcript": "Audio", "slide_ocr": "Slide Text", "slide_vision": "Slide Visuals"}[x]
    )

    if query:
        with st.spinner("Searching..."):
            results = search_talk_content(query, content_types=content_filter if content_filter else None, match_count=num_results)

        if results:
            st.markdown("### AI Answer")
            with st.spinner("Generating insights..."):
                insights = generate_search_insights(query, results)

            st.markdown(f"""<div class="summary-container">{insights}</div>""", unsafe_allow_html=True)

            st.markdown(f"### Source Segments ({len(results)} found)")

            for r in results:
                content_type_label = {
                    "audio_transcript": "Audio",
                    "slide_ocr": "Slide Text",
                    "slide_vision": "Slide Visual"
                }.get(r['content_type'], r['content_type'])

                slide_info = f" (Slide {r['slide_number']})" if r.get('slide_number') else ""

                with st.expander(f"{content_type_label}{slide_info} — {r['similarity']:.0%} match"):
                    st.markdown(f"> {r['content']}")
        else:
            st.warning("No results found. Try different keywords.")

elif st.session_state.active_view == "talk_detail" and st.session_state.selected_talk:
    talk = get_talk_by_id(st.session_state.selected_talk)

    if not talk:
        st.error("Talk not found")
        st.session_state.active_view = "talks"
        st.rerun()

    st.title(talk['title'])
    if talk.get('speaker'):
        st.caption(f"Speaker: {talk['speaker']}")

    # Stats
    chunks = get_talk_chunks(talk["id"])
    audio_chunks = [c for c in chunks if c['content_type'] == 'audio_transcript']
    slide_numbers = set([c['slide_number'] for c in chunks if c.get('slide_number')])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Audio Chunks", len(audio_chunks))
    with col2:
        st.metric("Slides", len(slide_numbers))
    with col3:
        total_chars = sum(len(c['content']) for c in chunks)
        st.metric("Total Characters", f"{total_chars:,}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([":material/upload: Upload", ":material/summarize: Summary", ":material/lightbulb: Insights", ":material/search: Search", ":material/download: Export", ":material/settings: Manage"])

    with tab1:
        st.markdown("### Upload Content")

        upload_type = st.radio("Content Type", ["Audio Recording", "Slide Images"], horizontal=True)

        if upload_type == "Audio Recording":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)

            uploaded_audio = st.file_uploader(
                "Choose audio file",
                type=["mp3", "mp4", "m4a", "wav", "webm", "mpeg", "mpga"],
                help="Supported: MP3, MP4, M4A, WAV, WebM"
            )

            model = st.selectbox("Transcription Model", AVAILABLE_MODELS, index=0)

            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_audio:
                if st.button("Transcribe Audio", type="primary", use_container_width=True, icon=":material/mic:"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp:
                        tmp.write(uploaded_audio.getvalue())
                        tmp_path = tmp.name

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(p, msg):
                        progress_bar.progress(p)
                        status_text.text(msg)

                    result = ingest_audio(tmp_path, uploaded_audio.name, talk["id"], model, update_progress)

                    os.unlink(tmp_path)

                    if result["status"] == "success":
                        st.success(result['messages'][-1])
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(result['messages'][-1])

        else:  # Slide Images
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)

            uploaded_images = st.file_uploader(
                "Choose slide images",
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                accept_multiple_files=True,
                help="Upload slides in order. Supported: PNG, JPG, WebP, HEIC"
            )

            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_images:
                st.markdown(f"**{len(uploaded_images)} slides selected**")

                if st.button("Process Slides", type="primary", use_container_width=True, icon=":material/image:"):
                    images_to_process = []
                    for uf in uploaded_images:
                        img = Image.open(io.BytesIO(uf.getvalue()))
                        images_to_process.append((img, uf.name))

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(p, msg):
                        progress_bar.progress(p)
                        status_text.text(msg)

                    result = ingest_images(images_to_process, talk["id"], update_progress)

                    if result["status"] == "success":
                        st.success(result['messages'][-1])
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(result['messages'][-1])

        # Previously uploaded files section
        st.divider()
        st.markdown("### Previously Uploaded")

        uploaded_files = get_uploaded_files(talk["id"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Audio Files**")
            if uploaded_files["audio"]:
                for f in uploaded_files["audio"]:
                    st.caption(f":material/mic: {f}")
            else:
                st.caption("No audio files uploaded")

        with col2:
            st.markdown("**Slide Images**")
            if uploaded_files["slides"]:
                for f in uploaded_files["slides"]:
                    st.caption(f":material/image: {f}")
            else:
                st.caption("No slides uploaded")

    with tab2:
        if not chunks:
            st.info("Upload audio or slides first to generate a summary.")
        else:
            summary_instructions = st.text_area(
                "Custom instructions (optional)",
                placeholder="e.g., Focus on serverless topics, keep it under 500 words...",
                key="summary_instructions",
                height=68
            )
            if st.button(f"Generate Summary ({st.session_state.selected_model})", type="primary", use_container_width=True, icon=":material/auto_awesome:"):
                with st.spinner("Generating comprehensive summary..."):
                    generate_talk_summary(talk["id"], st.session_state.selected_model, summary_instructions or None)
                st.rerun()

            # Load all saved summaries
            saved_summaries = get_all_ai_content(talk["id"], "summary")

            if saved_summaries:
                st.markdown(f"### Generated Summaries ({len(saved_summaries)})")
                for idx, item in enumerate(saved_summaries):
                    timestamp = item['created_at'][:16].replace('T', ' ')
                    with st.expander(f"{item['model_used']} — {timestamp}", expanded=(idx == 0)):
                        st.markdown(item['content'])
                        if st.button("Delete", key=f"del_summary_{item['id']}", icon=":material/delete:"):
                            delete_ai_content(item['id'])
                            st.rerun()

    with tab3:
        st.markdown("### Talk Insights")

        if not chunks:
            st.info("Upload audio or slides first to generate insights.")
        else:
            # Load saved chat history
            saved_chat = get_chat_history(talk["id"])
            if saved_chat and not st.session_state.chat_history:
                st.session_state.chat_history = saved_chat

            # Key Quotes Section
            st.markdown("#### Key Quotes & Highlights")
            quotes_instructions = st.text_area(
                "Custom instructions (optional)",
                placeholder="e.g., Focus on quotes about cost optimization...",
                key="quotes_instructions",
                height=68
            )
            if st.button("Extract Quotes", use_container_width=True, icon=":material/format_quote:"):
                with st.spinner("Extracting key quotes..."):
                    extract_key_quotes(talk["id"], st.session_state.selected_model, quotes_instructions or None)
                st.rerun()

            saved_quotes = get_all_ai_content(talk["id"], "quotes")
            if saved_quotes:
                for idx, item in enumerate(saved_quotes):
                    timestamp = item['created_at'][:16].replace('T', ' ')
                    with st.expander(f"{item['model_used']} — {timestamp}", expanded=(idx == 0)):
                        st.markdown(item['content'])
                        if st.button("Delete", key=f"del_quotes_{item['id']}", icon=":material/delete:"):
                            delete_ai_content(item['id'])
                            st.rerun()
            else:
                st.caption("Click 'Extract Quotes' to find memorable quotes and statistics from this talk.")

            st.divider()

            # Action Items Section
            st.markdown("#### Action Items & Recommendations")
            actions_instructions = st.text_area(
                "Custom instructions (optional)",
                placeholder="e.g., Focus on actions for a startup team...",
                key="actions_instructions",
                height=68
            )
            if st.button("Extract Actions", use_container_width=True, icon=":material/checklist:"):
                with st.spinner("Extracting action items..."):
                    extract_action_items(talk["id"], st.session_state.selected_model, actions_instructions or None)
                st.rerun()

            saved_actions = get_all_ai_content(talk["id"], "actions")
            if saved_actions:
                for idx, item in enumerate(saved_actions):
                    timestamp = item['created_at'][:16].replace('T', ' ')
                    with st.expander(f"{item['model_used']} — {timestamp}", expanded=(idx == 0)):
                        st.markdown(item['content'])
                        if st.button("Delete", key=f"del_actions_{item['id']}", icon=":material/delete:"):
                            delete_ai_content(item['id'])
                            st.rerun()
            else:
                st.caption("Click 'Extract Actions' to find tasks and recommendations from this talk.")

            st.divider()

            # Q&A Chat Section
            st.markdown("#### Ask Questions About This Talk")
            st.caption(f"Using: {st.session_state.selected_model}")

            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(msg["user"])
                with st.chat_message("assistant"):
                    st.write(msg["assistant"])

            # Chat input
            user_question = st.chat_input("Ask a question about this talk...")

            if user_question:
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_with_talk(talk["id"], user_question, st.session_state.chat_history, st.session_state.selected_model)
                    st.write(response)

                st.session_state.chat_history.append({
                    "user": user_question,
                    "assistant": response
                })

                # Save chat history to database
                save_chat_history(talk["id"], st.session_state.chat_history, st.session_state.selected_model)
                st.rerun()

            if st.session_state.chat_history:
                if st.button("Clear Chat History", type="secondary", icon=":material/delete:"):
                    st.session_state.chat_history = []
                    save_chat_history(talk["id"], [], st.session_state.selected_model)
                    st.rerun()

    with tab4:
        st.markdown("### Search This Talk")

        talk_query = st.text_input("Search query", placeholder="Search within this talk...", key="talk_search", label_visibility="collapsed")

        if talk_query:
            with st.spinner("Searching..."):
                talk_results = search_talk_content(talk_query, talk_id=talk["id"], match_count=10)

            if talk_results:
                for r in talk_results:
                    content_type_label = {
                        "audio_transcript": "Audio",
                        "slide_ocr": "Text",
                        "slide_vision": "Visual"
                    }.get(r['content_type'], "")

                    slide_info = f" Slide {r['slide_number']}" if r.get('slide_number') else ""

                    with st.expander(f"{content_type_label}{slide_info} — {r['similarity']:.0%}"):
                        st.markdown(f"> {r['content']}")
            else:
                st.warning("No matches found in this talk.")

    with tab5:
        st.markdown("### Export to Markdown")

        include_transcript = st.checkbox("Include full transcript", value=True)

        if st.button("Generate Export", use_container_width=True, icon=":material/file_download:"):
            with st.spinner("Generating export..."):
                md_content = export_to_markdown(talk["id"], include_transcript)

            st.download_button(
                "Download Markdown",
                md_content,
                file_name=f"{talk['title'][:30].replace(' ', '_')}_notes.md",
                mime="text/markdown",
                use_container_width=True
            )

            with st.expander("Preview"):
                st.markdown(md_content)

    with tab6:
        st.markdown("### Manage Talk")

        # Edit talk details
        st.markdown("**Edit Talk Details**")
        with st.form("edit_talk_form"):
            edit_title = st.text_input("Title", value=talk["title"])
            edit_speaker = st.text_input("Speaker", value=talk.get("speaker") or "")

            if st.form_submit_button("Save Changes", type="primary", use_container_width=True, icon=":material/save:"):
                if edit_title:
                    update_talk(talk["id"], edit_title, edit_speaker or None)
                    st.success("Talk updated")
                    st.rerun()
                else:
                    st.warning("Title is required")

        st.divider()

        # Danger zone
        st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
        st.markdown("**Delete this talk**")
        st.caption("This will permanently remove the talk and all associated content.")

        confirm = st.checkbox(f"I understand this will delete '{talk['title']}'")

        if st.button("Delete Talk", type="secondary", disabled=not confirm, icon=":material/delete_forever:"):
            delete_talk(talk["id"])
            st.success("Talk deleted")
            st.session_state.active_view = "talks"
            st.session_state.selected_talk = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
