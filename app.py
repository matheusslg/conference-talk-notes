import streamlit as st
from supabase import create_client
from google import genai
from openai import OpenAI
import anthropic
from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif
import os
import tempfile
import io
import json
import base64
import re
import subprocess
from datetime import datetime
import time
from google.genai import types

# Register HEIF/HEIC support with Pillow
pillow_heif.register_heif_opener()

# Page config
st.set_page_config(
    page_title="Conference Talk Notes",
    layout="wide"
)

# Constants (defined early for use in login page)
CURRENT_YEAR = datetime.now().year
DEFAULT_EVENT = f"AWS re:Invent {CURRENT_YEAR}"

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
    st.caption(DEFAULT_EVENT)
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

# Photo capture delay: typical delay between slide appearing and user taking photo
# User opens camera, waits for transitions, then snaps - usually 5-15 seconds
PHOTO_CAPTURE_DELAY_SECONDS = 10

# AI Alignment Prompt - used to match slides with audio transcription
AI_ALIGNMENT_PROMPT = """You are aligning presentation slides with an audio transcript.

## TRANSCRIPT (with timestamps)
{transcript_json}

## SLIDES (in presentation order)
{slides_json}

## TASK
Match each slide to the transcript segments where the speaker discusses that slide's content.

## RULES
1. Each transcript segment should be assigned to exactly ONE slide
2. Slides appear in sequential order - the speaker generally moves forward
3. A slide may have ZERO segments (quick transition, title slide)
4. A slide may have MANY segments (speaker dwells on complex content)
5. Look for keyword matches, topic continuity, transitions like "next slide", "as you can see"
6. If unsure, prefer assigning to earlier slides

## OUTPUT FORMAT (JSON array only, no markdown)
[
  {{"slide_number": 1, "segment_indices": [0, 1]}},
  {{"slide_number": 2, "segment_indices": [2, 3, 4]}},
  ...
]

Include ALL slides. Assign ALL transcript segments. Output valid JSON only."""

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

    # Enrich with segment count and first thumbnail
    for talk in talks:
        chunks = supabase.from_("talk_chunks").select("content_type, slide_thumbnail, slide_number").eq("talk_id", talk["id"]).order("slide_number").execute()
        chunk_data = chunks.data or []

        # Count aligned segments (new format) or fall back to legacy counts
        aligned = len([c for c in chunk_data if c["content_type"] == "aligned_segment"])
        legacy = len([c for c in chunk_data if c["content_type"] in ("audio_transcript", "slide_ocr", "slide_vision")])
        talk["segment_count"] = aligned or legacy

        # Get first thumbnail
        thumbnails = [c.get("slide_thumbnail") for c in chunk_data if c.get("slide_thumbnail")]
        talk["first_thumbnail"] = thumbnails[0] if thumbnails else None

        # Check if has summary (processed indicator)
        ai_content = supabase.from_("talk_ai_content").select("content_type").eq("talk_id", talk["id"]).execute()
        talk["has_summary"] = any(c["content_type"] == "summary" for c in (ai_content.data or []))

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

@st.dialog("Delete Talk")
def delete_talk_dialog(talk_id: str, talk_title: str):
    """Modal dialog for confirming talk deletion."""
    st.warning(f"This will permanently delete **{talk_title}** and all associated content. This cannot be undone.")
    st.markdown(f"Type **{talk_title}** to confirm:")
    confirm_name = st.text_input("Talk name", key="dialog_confirm_delete_name", label_visibility="collapsed")

    col_yes, col_no = st.columns(2)
    with col_yes:
        name_matches = confirm_name.strip() == talk_title
        if st.button("Delete", type="primary", use_container_width=True, disabled=not name_matches, icon=":material/delete_forever:"):
            delete_talk(talk_id)
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

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

# ============== Aligned Audio-Slide Processing ==============

def create_thumbnail_base64(image_bytes: bytes, max_size: int = 800) -> str:
    """Create a readable thumbnail and return as base64 string."""
    import base64
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_timestamp(image_bytes: bytes) -> datetime | None:
    """Extract DateTimeOriginal from image EXIF data."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_audio_creation_time(file_path: str) -> datetime | None:
    """Extract creation timestamp from audio file metadata using ffprobe.

    Works with M4A (iPhone voice memos), MP3, WAV, and other common formats.
    Returns None if no reliable timestamp can be determined.
    """
    try:
        # Use ffprobe to extract format metadata
        cmd = [
            'ffprobe', '-v', 'error',
            '-print_format', 'json',
            '-show_format',
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # ffprobe failed - can't determine creation time
            return None

        metadata = json.loads(result.stdout)
        tags = metadata.get('format', {}).get('tags', {})

        # Try common creation time fields (case-insensitive)
        tags_lower = {k.lower(): v for k, v in tags.items()}
        for field in ['creation_time', 'date', 'icrd', 'date_recorded']:
            if field in tags_lower:
                value = tags_lower[field]
                # Handle ISO format with timezone: "2024-12-04T14:00:00.000000Z"
                if 'T' in value:
                    # Parse with timezone handling
                    value = value.replace('Z', '+00:00')
                    # Remove microseconds if present
                    if '.' in value:
                        base = value.split('.')[0]
                        tz_part = value[value.rfind('+'):] if '+' in value else (value[value.rfind('-'):] if value.count('-') > 2 else '')
                        value = base + tz_part if tz_part else base
                    dt = datetime.fromisoformat(value)
                    # Convert to naive local time for comparison with EXIF (which is local)
                    if dt.tzinfo is not None:
                        dt = dt.astimezone().replace(tzinfo=None)
                    return dt
                # Handle simple date format: "2024-12-04"
                if len(value) == 10 and value.count('-') == 2:
                    return datetime.strptime(value, "%Y-%m-%d")

        # No embedded metadata found
        return None

    except Exception:
        return None


def transcribe_audio_with_timestamps(file_path: str) -> list[dict]:
    """
    Transcribe audio and return segments with timestamps.
    Returns: [{"start": "00:01:30", "end": "00:02:45", "text": "..."}, ...]
    """
    uploaded_file = ai.files.upload(file=file_path)

    # Wait for file to be processed
    import time
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = ai.files.get(name=uploaded_file.name)

    prompt = """Transcribe this audio with timestamps.

Output format (JSON array only, no markdown):
[
  {"start": "00:00:00", "end": "00:01:23", "text": "transcribed text here"},
  {"start": "00:01:23", "end": "00:02:45", "text": "next segment"},
  ...
]

Rules:
- Create segments of roughly 30-60 seconds each, breaking at natural pauses
- Include ALL spoken content
- Timestamps in HH:MM:SS format
- Output ONLY valid JSON array, no explanation or markdown"""

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt]
    )

    text = response.text.strip()
    # Clean up markdown if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)

def parse_timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def format_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return "??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ============== Step-by-Step Processing Functions ==============

def step_transcribe_audio(audio_bytes: bytes, audio_name: str) -> dict:
    """Step 1: Transcribe audio using Gemini.

    Returns dict with:
        - segments: list of {start, end, text}
        - messages: list of status messages
    """
    result = {"segments": [], "messages": []}

    # Write to temp file
    suffix = os.path.splitext(audio_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Transcribe using existing function
        segments = transcribe_audio_with_timestamps(tmp_path)
        result["segments"] = segments
        result["messages"].append(f"Transcribed {len(segments)} segments")

        # Calculate total duration
        if segments:
            last_end = parse_timestamp_to_seconds(segments[-1]["end"])
            result["messages"].append(f"Total duration: {format_seconds_to_timestamp(last_end)}")
    finally:
        os.unlink(tmp_path)

    return result


def step_prepare_slides(slide_files: list) -> dict:
    """Step 2a: Extract EXIF timestamps from slides and sort.

    Args:
        slide_files: list of dicts with {bytes, name}

    Returns dict with:
        - slides_with_time: sorted list of slide dicts
        - use_exif_alignment: bool
        - messages: list of status messages
    """
    result = {"slides_with_time": [], "use_exif_alignment": False, "messages": []}

    for slide in slide_files:
        img_bytes = slide["bytes"]
        timestamp = get_image_timestamp(img_bytes)
        result["slides_with_time"].append({
            "bytes": img_bytes,
            "timestamp": timestamp,
            "name": slide["name"],
            "relative_seconds": None
        })

    # Sort by timestamp (None timestamps go to end, then by filename)
    result["slides_with_time"].sort(
        key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.max, x["name"])
    )

    # Calculate relative timestamps for slides with EXIF
    recording_start = result["slides_with_time"][0]["timestamp"] if result["slides_with_time"][0]["timestamp"] else None
    for slide in result["slides_with_time"]:
        if slide["timestamp"] and recording_start:
            delta = slide["timestamp"] - recording_start
            slide["relative_seconds"] = delta.total_seconds()

    # Decide alignment strategy: EXIF if ≥50% have timestamps
    slides_with_exif = sum(1 for s in result["slides_with_time"] if s["relative_seconds"] is not None)
    result["use_exif_alignment"] = slides_with_exif >= len(result["slides_with_time"]) * 0.5

    # Debug logging
    first_slide_ts = result["slides_with_time"][0]["timestamp"]
    if first_slide_ts:
        result["messages"].append(f"First slide taken at: {first_slide_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        result["messages"].append("No EXIF timestamp found in first slide")

    result["messages"].append(f"Slides with EXIF: {slides_with_exif}/{len(result['slides_with_time'])}")
    result["messages"].append(f"Alignment mode: {'EXIF-based' if result['use_exif_alignment'] else 'AI-based'}")

    return result


def step_upload_audio(audio_bytes: bytes, audio_name: str, talk_id: str) -> dict:
    """Step 2b: Upload audio to Supabase storage and extract creation time.

    Returns dict with:
        - audio_url: str or None
        - audio_start_time: datetime or None
        - messages: list of status messages
    """
    result = {"audio_url": None, "audio_start_time": None, "messages": []}

    # Write to temp file for metadata extraction
    suffix = os.path.splitext(audio_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Extract audio creation time
        result["audio_start_time"] = get_audio_creation_time(tmp_path)
        if result["audio_start_time"]:
            result["messages"].append(f"Audio recording started at: {result['audio_start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            result["messages"].append("No audio creation time found in metadata")
    finally:
        os.unlink(tmp_path)

    # Upload to Supabase Storage
    try:
        audio_filename = f"{talk_id}/{audio_name}"
        supabase.storage.from_("talk-audio").upload(
            audio_filename,
            audio_bytes,
            {"content-type": "audio/mpeg"}
        )
        result["audio_url"] = supabase.storage.from_("talk-audio").get_public_url(audio_filename)
        result["messages"].append("Audio uploaded for playback")

        # Update talk record
        update_data = {"audio_url": result["audio_url"]}
        if result["audio_start_time"]:
            update_data["audio_start_timestamp"] = result["audio_start_time"].isoformat()
        supabase.from_("talks").update(update_data).eq("id", talk_id).execute()

    except Exception as upload_error:
        result["messages"].append(f"Audio upload skipped: {str(upload_error)}")

    return result


def step_align_slides_to_audio(
    slides_with_time: list,
    audio_segments: list,
    audio_start_time,
    use_exif: bool
) -> dict:
    """Step 3: Align slides to audio segments.

    Returns dict with:
        - alignment: list of aligned slide dicts
        - messages: list of status messages
    """
    result = {"alignment": [], "messages": []}

    # Recalculate relative timestamps using audio start time (if available)
    if audio_start_time and slides_with_time:
        for slide in slides_with_time:
            if slide["timestamp"]:
                delta = slide["timestamp"] - audio_start_time
                slide["relative_seconds"] = delta.total_seconds()
            else:
                slide["relative_seconds"] = None

        # Re-check EXIF alignment availability
        slides_with_exif = sum(1 for s in slides_with_time if s["relative_seconds"] is not None)
        use_exif = slides_with_exif >= len(slides_with_time) * 0.5

        # Debug: log alignment offset
        first_slide = slides_with_time[0]
        if first_slide["relative_seconds"] is not None:
            offset = first_slide["relative_seconds"]
            result["messages"].append(f"First slide offset from audio start: {offset:.1f}s ({format_seconds_to_timestamp(offset)})")

    if use_exif:
        # EXIF-based alignment
        result["messages"].append("Using EXIF-based alignment")
        result["messages"].append(f"Photo capture delay adjustment: -{PHOTO_CAPTURE_DELAY_SECONDS}s")

        for idx, slide in enumerate(slides_with_time):
            img = Image.open(io.BytesIO(slide["bytes"]))
            ocr_text = extract_slide_ocr(img)
            vision_desc = describe_slide_vision(img)
            thumbnail = create_thumbnail_base64(slide["bytes"])

            next_slide_time = slides_with_time[idx + 1]["relative_seconds"] if idx + 1 < len(slides_with_time) else None
            matched_audio = match_audio_to_slide(audio_segments, slide["relative_seconds"], next_slide_time)

            # Calculate adjusted times for storage (matching what match_audio_to_slide uses)
            adjusted_start = max(0, slide["relative_seconds"] - PHOTO_CAPTURE_DELAY_SECONDS) if slide["relative_seconds"] is not None else None
            adjusted_end = (next_slide_time - PHOTO_CAPTURE_DELAY_SECONDS) if next_slide_time is not None else None

            result["alignment"].append({
                "slide_number": idx + 1,
                "slide_name": slide["name"],
                "thumbnail": thumbnail,
                "start_time_seconds": adjusted_start,
                "end_time_seconds": adjusted_end,
                "matched_audio": matched_audio,
                "ocr_text": ocr_text,
                "vision_description": vision_desc
            })
    else:
        # AI-based alignment
        result["messages"].append("Using AI-based alignment (insufficient EXIF timestamps)")

        # First process all slides
        processed_slides = []
        for idx, slide in enumerate(slides_with_time):
            img = Image.open(io.BytesIO(slide["bytes"]))
            ocr_text = extract_slide_ocr(img)
            vision_desc = describe_slide_vision(img)
            thumbnail = create_thumbnail_base64(slide["bytes"])

            processed_slides.append({
                "slide_number": idx + 1,
                "ocr_text": ocr_text,
                "vision_description": vision_desc,
                "thumbnail": thumbnail,
                "filename": slide["name"]
            })

        # Run AI alignment
        try:
            alignments = align_slides_with_ai(audio_segments, processed_slides)
            result["messages"].append("AI alignment successful")
        except Exception as align_error:
            result["messages"].append(f"AI alignment failed, using sequential fallback: {str(align_error)}")
            alignments = fallback_sequential_alignment(audio_segments, len(processed_slides))

        # Build alignment result
        for alignment in alignments:
            slide_idx = alignment["slide_number"] - 1
            slide = processed_slides[slide_idx]

            result["alignment"].append({
                "slide_number": alignment["slide_number"],
                "slide_name": slide["filename"],
                "thumbnail": slide["thumbnail"],
                "start_time_seconds": alignment["start_time_seconds"],
                "end_time_seconds": alignment["end_time_seconds"],
                "matched_audio": alignment["matched_audio"],
                "ocr_text": slide["ocr_text"],
                "vision_description": slide["vision_description"]
            })

    result["messages"].append(f"Aligned {len(result['alignment'])} slides")
    return result


def step_store_segments(talk_id: str, alignment: list, audio_name: str = "") -> dict:
    """Step 4: Generate embeddings and store aligned segments.

    Returns dict with:
        - segment_count: int
        - messages: list of status messages
    """
    result = {"segment_count": 0, "messages": []}

    for item in alignment:
        time_str = format_seconds_to_timestamp(item["start_time_seconds"])
        aligned_content = f"""## Slide {item['slide_number']} [{time_str}]

### Slide Text
{item['ocr_text'] if item['ocr_text'] else "[No text detected]"}

### Visual Description
{item['vision_description']}

### Speaker Said
{item['matched_audio']}
"""

        embedding = generate_embedding(aligned_content)

        source_file = f"{audio_name}+{item['slide_name']}" if audio_name else item['slide_name']

        supabase.from_("talk_chunks").insert({
            "talk_id": talk_id,
            "content": aligned_content,
            "content_type": "aligned_segment",
            "source_file": source_file,
            "slide_number": item["slide_number"],
            "chunk_index": 0,
            "start_time_seconds": item["start_time_seconds"],
            "end_time_seconds": item["end_time_seconds"],
            "embedding": embedding,
            "slide_thumbnail": item["thumbnail"]
        }).execute()

        result["segment_count"] += 1

    result["messages"].append(f"Stored {result['segment_count']} aligned segments")
    return result


def reset_processing_state():
    """Clear all processing-related session state."""
    keys_to_clear = [
        'processing_step', 'pending_talk_id', 'pending_audio_file',
        'pending_slide_files', 'transcript_segments', 'alignment_result',
        'slides_with_time', 'audio_start_time', 'audio_url', 'use_exif',
        'multimodal_result', 'multimodal_messages', 'multimodal_model_used',
        'balloons_shown', 'final_segment_count', 'store_messages'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ============== Unified Multimodal Processing ==============

# Maximum audio file size before chunking (in bytes) - ~20MB
MAX_AUDIO_SIZE_BYTES = 20 * 1024 * 1024


def build_multimodal_prompt(slides_metadata: list) -> str:
    """Build the prompt for unified multimodal processing."""
    prompt = """You are analyzing a conference presentation. You have:
1. An audio recording of the speaker
2. Photos of each slide (in chronological order)
3. Metadata about when each photo was taken

IMPORTANT CONTEXT:
- Photos were taken 5-15 seconds AFTER each slide appeared on screen
- The speaker discusses the slide content while showing it
- Use semantic understanding to match what's being said to what's on screen
- Listen to the FULL audio and map it to the slides based on content

SLIDE METADATA:
"""
    for slide in slides_metadata:
        rel_time = slide.get('relative_seconds')
        if rel_time is not None:
            prompt += f"- Slide {slide['number']}: {slide['name']}, photo taken at ~{rel_time:.0f}s into the talk\n"
        else:
            prompt += f"- Slide {slide['number']}: {slide['name']}\n"

    prompt += """

OUTPUT FORMAT (JSON array, one entry per slide):
[
  {
    "slide_number": 1,
    "ocr_text": "All visible text on the slide exactly as shown",
    "visual_description": "Description of diagrams, charts, icons, and visual elements",
    "transcript_text": "What the speaker said while discussing this slide (full transcript for this slide)",
    "start_time_seconds": 0.0,
    "end_time_seconds": 120.5,
    "key_points": ["Main takeaway 1", "Main takeaway 2"]
  }
]

RULES:
- transcript_text must capture the FULL speaker discussion for each slide
- Times are in seconds from audio start
- First slide starts at 0
- Last slide ends when audio ends
- Slides must NOT have overlapping time ranges
- Each second of audio should be assigned to exactly ONE slide
- Output ONLY valid JSON array, no markdown wrapper or explanation
- If a slide has no discernible text, set ocr_text to empty string
- If a slide has no visual elements (text only), set visual_description to empty string
"""
    return prompt


def execute_multimodal_request(
    audio_file,
    slide_images: list,
    slides_metadata: list,
    model: str = "gemini-2.5-pro"
) -> list:
    """Execute unified multimodal request to Gemini.

    Args:
        audio_file: Uploaded Gemini file reference
        slide_images: List of image bytes
        slides_metadata: List of slide metadata dicts
        model: Gemini model to use

    Returns:
        List of alignment dicts from Gemini
    """
    contents = []

    # Add audio file first
    contents.append(audio_file)

    # Supported image MIME types for Gemini
    SUPPORTED_IMAGE_TYPES = {"jpeg", "png", "gif", "webp"}

    # Add each slide image with label
    for i, img_bytes in enumerate(slide_images):
        contents.append(f"[Slide {i+1}]")

        # Open and normalize image format
        img = Image.open(io.BytesIO(img_bytes))
        fmt = (img.format or "JPEG").lower()

        # Convert unsupported formats (MPO, HEIC, etc.) to JPEG
        if fmt not in SUPPORTED_IMAGE_TYPES:
            # Convert to RGB (handles RGBA, palette modes, etc.)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            img_bytes = buffer.getvalue()
            mime_type = "image/jpeg"
        else:
            mime_type = f"image/{fmt}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    # Add the prompt
    prompt = build_multimodal_prompt(slides_metadata)
    contents.append(prompt)

    response = ai.models.generate_content(
        model=model,
        contents=contents
    )

    # Parse JSON response
    text = response.text.strip()
    # Clean up markdown if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)


def get_gemini_model_for_multimodal(selected_model: str) -> str:
    """Get the appropriate Gemini model for multimodal processing.

    Only Gemini supports unified audio+vision processing.
    If a non-Gemini model is selected, returns a sensible Gemini default.

    Args:
        selected_model: The model selected in the UI dropdown

    Returns:
        A Gemini model name suitable for multimodal processing
    """
    if selected_model.startswith("gemini"):
        return selected_model
    # Default to gemini-2.5-pro for non-Gemini models
    return "gemini-2.5-pro"


def step_process_multimodal(
    audio_file: dict,
    slide_files: list,
    talk_id: str,
    model: str = "gemini-2.5-pro"
) -> dict:
    """Unified processing step - sends audio + all slides to Gemini in one request.

    Args:
        audio_file: dict with {bytes, name}
        slide_files: list of dicts with {bytes, name}
        talk_id: Talk ID for audio upload
        model: Gemini model to use (must be a Gemini model)

    Returns:
        dict with:
            - alignment: list of slide alignment dicts
            - messages: list of status messages
            - audio_url: URL of uploaded audio (for playback)
            - model_used: The model that was used for processing
    """
    # Ensure we have a valid Gemini model
    gemini_model = get_gemini_model_for_multimodal(model)

    result = {"alignment": [], "messages": [], "audio_url": None, "model_used": gemini_model}

    # 1. Extract EXIF timestamps from slides and sort
    slides_with_time = []
    for slide in slide_files:
        timestamp = get_image_timestamp(slide["bytes"])
        slides_with_time.append({
            "bytes": slide["bytes"],
            "timestamp": timestamp,
            "name": slide["name"],
            "relative_seconds": None
        })

    # Sort by timestamp (None timestamps go to end, then by filename)
    slides_with_time.sort(
        key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.max, x["name"])
    )

    # Calculate relative timestamps
    if slides_with_time and slides_with_time[0]["timestamp"]:
        recording_start = slides_with_time[0]["timestamp"]
        for slide in slides_with_time:
            if slide["timestamp"]:
                delta = slide["timestamp"] - recording_start
                slide["relative_seconds"] = delta.total_seconds()

    slides_with_exif = sum(1 for s in slides_with_time if s["timestamp"] is not None)
    result["messages"].append(f"Slides with EXIF timestamps: {slides_with_exif}/{len(slides_with_time)}")

    # 2. Upload audio to Gemini for processing
    suffix = os.path.splitext(audio_file['name'])[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file['bytes'])
        tmp_path = tmp.name

    try:
        result["messages"].append("Uploading audio to Gemini...")
        uploaded_audio = ai.files.upload(file=tmp_path)

        # Wait for processing
        while uploaded_audio.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_audio = ai.files.get(name=uploaded_audio.name)

        if uploaded_audio.state.name == "FAILED":
            raise Exception("Audio processing failed in Gemini")

        result["messages"].append("Audio processed, running unified analysis...")

        # 3. Prepare slides metadata for prompt
        slides_metadata = [
            {
                "number": i + 1,
                "name": s["name"],
                "relative_seconds": s.get("relative_seconds")
            }
            for i, s in enumerate(slides_with_time)
        ]

        # 4. Execute multimodal request
        slide_images = [s["bytes"] for s in slides_with_time]
        alignment_result = execute_multimodal_request(
            uploaded_audio,
            slide_images,
            slides_metadata,
            model=gemini_model
        )

        result["messages"].append(f"Gemini returned {len(alignment_result)} slide analyses")

        # 5. Add thumbnails and slide names to results
        for i, item in enumerate(alignment_result):
            item["thumbnail"] = create_thumbnail_base64(slides_with_time[i]["bytes"])
            item["slide_name"] = slides_with_time[i]["name"]

        result["alignment"] = alignment_result

    finally:
        os.unlink(tmp_path)

    # 6. Upload audio to Supabase Storage for playback
    try:
        audio_filename = f"{talk_id}/{audio_file['name']}"
        supabase.storage.from_("talk-audio").upload(
            audio_filename,
            audio_file['bytes'],
            {"content-type": "audio/mpeg"}
        )
        result["audio_url"] = supabase.storage.from_("talk-audio").get_public_url(audio_filename)
        result["messages"].append("Audio uploaded for playback")

        # Update talk record
        supabase.from_("talks").update({"audio_url": result["audio_url"]}).eq("id", talk_id).execute()

    except Exception as upload_error:
        result["messages"].append(f"Audio storage skipped: {str(upload_error)}")

    # 7. Upload slides to Supabase Storage for multimodal summary/insights
    try:
        slide_urls = []
        for i, slide in enumerate(slides_with_time):
            slide_filename = f"{talk_id}/slide_{i+1:03d}.jpg"

            # Normalize image to JPEG for storage
            img = Image.open(io.BytesIO(slide["bytes"]))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            normalized_bytes = buffer.getvalue()

            supabase.storage.from_("talk-slides").upload(
                slide_filename,
                normalized_bytes,
                {"content-type": "image/jpeg"}
            )

            url = supabase.storage.from_("talk-slides").get_public_url(slide_filename)
            slide_urls.append(url)

        # Store slide URLs in talk record
        supabase.from_("talks").update({"slide_urls": slide_urls}).eq("id", talk_id).execute()
        result["slide_urls"] = slide_urls
        result["messages"].append(f"Uploaded {len(slide_urls)} slides for multimodal")

    except Exception as slide_error:
        result["messages"].append(f"Slide storage skipped: {str(slide_error)}")

    return result


def step_store_multimodal_results(talk_id: str, alignment: list, audio_name: str = "") -> dict:
    """Store multimodal processing results to database.

    Args:
        talk_id: Talk ID
        alignment: List of slide alignment dicts from Gemini
        audio_name: Original audio filename

    Returns:
        dict with:
            - segment_count: number of segments stored
            - messages: list of status messages
    """
    result = {"segment_count": 0, "messages": []}

    for item in alignment:
        # Combine content for storage and embedding
        key_points_text = "\n".join(f"- {p}" for p in item.get("key_points", []))

        combined_content = f"""## Slide {item['slide_number']} [{format_seconds_to_timestamp(item.get('start_time_seconds'))}]

### Slide Text
{item.get('ocr_text') or '[No text detected]'}

### Visual Description
{item.get('visual_description') or '[Text-only slide]'}

### Speaker Said
{item.get('transcript_text') or '[No transcript]'}

### Key Points
{key_points_text or '[No key points]'}
"""

        embedding = generate_embedding(combined_content)

        source_file = f"{audio_name}+{item['slide_name']}" if audio_name else item['slide_name']

        supabase.from_("talk_chunks").insert({
            "talk_id": talk_id,
            "content": combined_content,
            "content_type": "aligned_segment",
            "source_file": source_file,
            "slide_number": item["slide_number"],
            "chunk_index": 0,
            "start_time_seconds": item.get("start_time_seconds"),
            "end_time_seconds": item.get("end_time_seconds"),
            "embedding": embedding,
            "slide_thumbnail": item.get("thumbnail")
        }).execute()

        result["segment_count"] += 1

    result["messages"].append(f"Stored {result['segment_count']} aligned segments")
    return result


def parse_aligned_content(content: str) -> dict:
    """Parse the aligned segment content into sections."""
    sections = {
        'slide_text': '',
        'visual_desc': '',
        'audio': ''
    }

    current_section = None
    lines = content.split('\n')

    for line in lines:
        if '### Slide Text' in line:
            current_section = 'slide_text'
        elif '### Visual Description' in line:
            current_section = 'visual_desc'
        elif '### Speaker Audio' in line or '### Speaker Said' in line:
            current_section = 'audio'
        elif current_section and line.strip() and not line.startswith('##'):
            sections[current_section] += line + '\n'

    # Patterns that indicate no audio content
    NO_AUDIO_PATTERNS = [
        '[No matching audio in this time range]',
        '[No timestamp - audio not aligned]',
        '[Quick transition slide - no extended discussion]',
        '[No audio segment assigned]',
    ]

    # Clean up
    for key in sections:
        sections[key] = sections[key].strip()
        if sections[key] == '[No text detected]':
            sections[key] = ''
        # Normalize all no-audio patterns to a single marker
        for pattern in NO_AUDIO_PATTERNS:
            if pattern in sections[key]:
                sections[key] = '_No matching audio_'
                break

    return sections

def format_timestamped_segments(segments: list) -> str:
    """Format audio segments with timestamps for storage and display."""
    lines = []
    for seg in segments:
        start_seconds = parse_timestamp_to_seconds(seg["start"])
        timestamp = format_seconds_to_timestamp(start_seconds)
        lines.append(f"[{timestamp}] {seg['text']}")
    return "\n".join(lines)


def match_audio_to_slide(audio_segments: list, slide_start: float, next_slide_time: float) -> str:
    """Find audio segments that belong to this slide's discussion.

    IMPORTANT: Photo timestamps are typically 5-15 seconds AFTER the slide appeared,
    because the user opens the camera and waits for transitions before taking the photo.
    We shift backwards by PHOTO_CAPTURE_DELAY_SECONDS to capture the full discussion.

    Each segment is assigned to exactly ONE slide based on where it starts,
    preventing duplicate transcripts across slides.
    """
    if slide_start is None:
        return "[No timestamp - audio not aligned]"

    # Shift backwards to account for photo capture delay
    adjusted_start = max(0, slide_start - PHOTO_CAPTURE_DELAY_SECONDS)
    adjusted_end = (next_slide_time - PHOTO_CAPTURE_DELAY_SECONDS) if next_slide_time else None

    matched_segments = []
    for seg in audio_segments:
        seg_start = parse_timestamp_to_seconds(seg["start"])

        if adjusted_end is not None:
            # Segment STARTS within this slide's adjusted interval
            if adjusted_start <= seg_start < adjusted_end:
                matched_segments.append(seg)
        else:
            # Last slide - gets all remaining segments that start at or after adjusted_start
            if seg_start >= adjusted_start:
                matched_segments.append(seg)

    if not matched_segments:
        return "[No matching audio in this time range]"

    return format_timestamped_segments(matched_segments)

def align_slides_with_ai(audio_segments: list, slides_data: list) -> list:
    """
    Use Gemini to semantically align slides with transcript segments.
    Returns list of alignment results with matched audio text and timestamps.
    """
    # Prepare transcript for prompt (truncate long segments)
    transcript_for_prompt = [
        {
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"][:500]
        }
        for i, seg in enumerate(audio_segments)
    ]

    # Prepare slides for prompt
    slides_for_prompt = [
        {
            "slide_number": i + 1,
            "text": slide.get("ocr_text", "")[:300],
            "visual": slide.get("vision_description", "")[:200]
        }
        for i, slide in enumerate(slides_data)
    ]

    prompt = AI_ALIGNMENT_PROMPT.format(
        transcript_json=json.dumps(transcript_for_prompt, indent=2),
        slides_json=json.dumps(slides_for_prompt, indent=2)
    )

    response = ai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Parse JSON response
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    alignment = json.loads(text)

    # Validate and deduplicate: ensure each segment is assigned to exactly one slide
    used_indices: set[int] = set()
    num_segments = len(audio_segments)

    # Convert to usable format with timestamps
    results = []
    for item in alignment:
        slide_num = item["slide_number"]
        seg_indices = item.get("segment_indices", [])

        # Validate: filter out-of-bounds indices and already-used indices
        seg_indices = [i for i in seg_indices if 0 <= i < num_segments and i not in used_indices]

        # Track used indices to prevent duplicates
        used_indices.update(seg_indices)

        if seg_indices:
            start_time = parse_timestamp_to_seconds(audio_segments[seg_indices[0]]["start"])
            end_time = parse_timestamp_to_seconds(audio_segments[seg_indices[-1]]["end"])
            matched_segments = [audio_segments[i] for i in seg_indices]
            matched_text = format_timestamped_segments(matched_segments)
        else:
            start_time = None
            end_time = None
            matched_text = "[Quick transition slide - no extended discussion]"

        results.append({
            "slide_number": slide_num,
            "segment_indices": seg_indices,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "matched_audio": matched_text
        })

    return results

def fallback_sequential_alignment(audio_segments: list, num_slides: int) -> list:
    """
    Distribute audio segments evenly across slides as a fallback.
    Used when AI alignment fails. Each segment is assigned to exactly one slide.
    """
    if not audio_segments or num_slides == 0:
        return []

    # Calculate even distribution with remainder handling
    total_segments = len(audio_segments)
    segments_per_slide = total_segments // num_slides
    remainder = total_segments % num_slides

    results = []
    current_idx = 0

    for slide_num in range(1, num_slides + 1):
        # Give extra segments to early slides to handle remainder evenly
        count = segments_per_slide + (1 if slide_num <= remainder else 0)
        indices = list(range(current_idx, current_idx + count))

        if indices:
            start_time = parse_timestamp_to_seconds(audio_segments[indices[0]]["start"])
            end_time = parse_timestamp_to_seconds(audio_segments[indices[-1]]["end"])
            matched_segments = [audio_segments[j] for j in indices]
            matched_text = format_timestamped_segments(matched_segments)
        else:
            start_time = None
            end_time = None
            matched_text = "[No audio segment assigned]"

        results.append({
            "slide_number": slide_num,
            "segment_indices": indices,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "matched_audio": matched_text
        })

        current_idx += count  # Move to next batch (no overlap possible)

    return results

def process_talk_content(talk_id: str, audio_file=None, slide_files: list = None, progress_callback=None) -> dict:
    """Process audio and/or slides. Handles audio-only, slides-only, or both together."""
    results = {"status": "success", "messages": [], "segments": 0}

    if not audio_file and not slide_files:
        results["status"] = "error"
        results["messages"].append("No content provided")
        return results

    try:
        audio_segments = []
        slides_with_time = []
        use_exif_alignment = False
        slides_with_exif = 0

        # Process slides if provided - extract EXIF timestamps
        if slide_files:
            if progress_callback:
                progress_callback(0.05, "Preparing slides...")

            for slide in slide_files:
                img_bytes = slide.getvalue()
                timestamp = get_image_timestamp(img_bytes)
                slides_with_time.append({
                    "file": slide,
                    "bytes": img_bytes,
                    "timestamp": timestamp,
                    "name": slide.name
                })

            # Sort by timestamp (None timestamps go to end, then by filename)
            slides_with_time.sort(key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.max, x["name"]))

            # Calculate relative timestamps for slides with EXIF
            recording_start = slides_with_time[0]["timestamp"] if slides_with_time[0]["timestamp"] else None
            for slide in slides_with_time:
                if slide["timestamp"] and recording_start:
                    delta = slide["timestamp"] - recording_start
                    slide["relative_seconds"] = delta.total_seconds()
                else:
                    slide["relative_seconds"] = None

            # Decide alignment strategy: EXIF if ≥50% have timestamps, else AI
            slides_with_exif = sum(1 for s in slides_with_time if s["relative_seconds"] is not None)
            use_exif_alignment = slides_with_exif >= len(slides_with_time) * 0.5

            # Debug logging for slide timestamps
            first_slide_ts = slides_with_time[0]["timestamp"]
            if first_slide_ts:
                results["messages"].append(f"First slide taken at: {first_slide_ts.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                results["messages"].append("No EXIF timestamp found in first slide")

        # Transcribe audio if provided
        audio_start_time = None
        audio_url = None

        if audio_file:
            if progress_callback:
                progress_callback(0.15, "Transcribing audio with timestamps...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = tmp.name

            try:
                # Extract audio creation time for accurate alignment
                audio_start_time = get_audio_creation_time(tmp_path)
                if audio_start_time:
                    results["messages"].append(f"Audio recording started at: {audio_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    results["messages"].append("No audio creation time found in metadata - will use first slide as reference")

                # Transcribe the audio
                audio_segments = transcribe_audio_with_timestamps(tmp_path)
                results["messages"].append(f"Transcribed {len(audio_segments)} audio segments")

                # Upload audio to Supabase Storage for playback
                if progress_callback:
                    progress_callback(0.25, "Uploading audio for playback...")

                try:
                    audio_filename = f"{talk_id}/{audio_file.name}"
                    # Upload to storage bucket
                    supabase.storage.from_("talk-audio").upload(
                        audio_filename,
                        audio_file.getvalue(),
                        {"content-type": "audio/mpeg"}
                    )
                    # Get public URL
                    audio_url = supabase.storage.from_("talk-audio").get_public_url(audio_filename)
                    results["messages"].append("Audio uploaded for playback")

                    # Update talk record with audio info
                    update_data = {"audio_url": audio_url}
                    if audio_start_time:
                        update_data["audio_start_timestamp"] = audio_start_time.isoformat()
                    supabase.from_("talks").update(update_data).eq("id", talk_id).execute()

                except Exception as upload_error:
                    # Non-fatal: continue without audio playback
                    results["messages"].append(f"Audio upload skipped: {str(upload_error)}")

            finally:
                os.unlink(tmp_path)

        # Recalculate relative timestamps using audio start time (if available)
        if audio_start_time and slides_with_time:
            for slide in slides_with_time:
                if slide["timestamp"]:
                    delta = slide["timestamp"] - audio_start_time
                    slide["relative_seconds"] = delta.total_seconds()
                else:
                    slide["relative_seconds"] = None

            # Re-check EXIF alignment availability
            slides_with_exif = sum(1 for s in slides_with_time if s["relative_seconds"] is not None)
            use_exif_alignment = slides_with_exif >= len(slides_with_time) * 0.5

            # Debug: log alignment offset
            first_slide = slides_with_time[0]
            if first_slide["relative_seconds"] is not None:
                offset = first_slide["relative_seconds"]
                results["messages"].append(f"First slide offset from audio start: {offset:.1f}s ({format_seconds_to_timestamp(offset)})")

        # Case 1: Both audio and slides - create aligned segments (EXIF or AI)
        if audio_file and slide_files:
            total_slides = len(slides_with_time)

            if use_exif_alignment:
                # EXIF-based alignment: use photo timestamps to match audio
                if progress_callback:
                    progress_callback(0.3, f"Using EXIF timestamps ({slides_with_exif}/{total_slides} slides)...")

                for idx, slide in enumerate(slides_with_time):
                    if progress_callback:
                        progress_callback(0.3 + (0.6 * idx / total_slides), f"Processing slide {idx + 1}/{total_slides}...")

                    img = Image.open(io.BytesIO(slide["bytes"]))
                    ocr_text = extract_slide_ocr(img)
                    vision_desc = describe_slide_vision(img)
                    thumbnail = create_thumbnail_base64(slide["bytes"])

                    next_slide_time = slides_with_time[idx + 1]["relative_seconds"] if idx + 1 < len(slides_with_time) else None
                    matched_audio = match_audio_to_slide(audio_segments, slide["relative_seconds"], next_slide_time)

                    time_str = format_seconds_to_timestamp(slide["relative_seconds"])
                    aligned_content = f"""## Slide {idx + 1} [{time_str}]

### Slide Text
{ocr_text if ocr_text else "[No text detected]"}

### Visual Description
{vision_desc}

### Speaker Said
{matched_audio}
"""

                    embedding = generate_embedding(aligned_content)

                    supabase.from_("talk_chunks").insert({
                        "talk_id": talk_id,
                        "content": aligned_content,
                        "content_type": "aligned_segment",
                        "source_file": f"{audio_file.name}+{slide['name']}",
                        "slide_number": idx + 1,
                        "chunk_index": 0,
                        "start_time_seconds": slide["relative_seconds"],
                        "end_time_seconds": next_slide_time,
                        "embedding": embedding,
                        "slide_thumbnail": thumbnail
                    }).execute()

                    results["segments"] += 1

                results["messages"].append(f"Used EXIF timestamps for alignment ({slides_with_exif}/{total_slides} slides)")

            else:
                # AI-based alignment: use Gemini to match slides to audio
                if progress_callback:
                    progress_callback(0.3, "Processing slides with OCR and vision...")

                processed_slides = []
                for idx, slide in enumerate(slides_with_time):
                    if progress_callback:
                        progress_callback(0.3 + (0.2 * idx / total_slides), f"Processing slide {idx + 1}/{total_slides}...")

                    img = Image.open(io.BytesIO(slide["bytes"]))
                    ocr_text = extract_slide_ocr(img)
                    vision_desc = describe_slide_vision(img)
                    thumbnail = create_thumbnail_base64(slide["bytes"])

                    processed_slides.append({
                        "slide_number": idx + 1,
                        "ocr_text": ocr_text,
                        "vision_description": vision_desc,
                        "thumbnail": thumbnail,
                        "filename": slide["name"]
                    })

                if progress_callback:
                    progress_callback(0.55, "Aligning slides to audio with AI...")

                try:
                    alignments = align_slides_with_ai(audio_segments, processed_slides)
                    results["messages"].append("Used AI-based alignment (no EXIF timestamps found)")
                except Exception as align_error:
                    results["messages"].append(f"AI alignment failed, using sequential fallback: {str(align_error)}")
                    alignments = fallback_sequential_alignment(audio_segments, len(processed_slides))

                if progress_callback:
                    progress_callback(0.7, "Storing aligned segments...")

                for alignment in alignments:
                    slide_idx = alignment["slide_number"] - 1
                    slide = processed_slides[slide_idx]

                    time_str = format_seconds_to_timestamp(alignment["start_time_seconds"])
                    aligned_content = f"""## Slide {alignment['slide_number']} [{time_str}]

### Slide Text
{slide['ocr_text'] if slide['ocr_text'] else "[No text detected]"}

### Visual Description
{slide['vision_description']}

### Speaker Said
{alignment['matched_audio']}
"""

                    embedding = generate_embedding(aligned_content)

                    supabase.from_("talk_chunks").insert({
                        "talk_id": talk_id,
                        "content": aligned_content,
                        "content_type": "aligned_segment",
                        "source_file": f"{audio_file.name}+{slide['filename']}",
                        "slide_number": alignment["slide_number"],
                        "chunk_index": 0,
                        "start_time_seconds": alignment["start_time_seconds"],
                        "end_time_seconds": alignment["end_time_seconds"],
                        "embedding": embedding,
                        "slide_thumbnail": slide["thumbnail"]
                    }).execute()

                    results["segments"] += 1

        # Case 2: Audio only - create segments from transcription
        elif audio_file and not slide_files:
            total_segments = len(audio_segments)
            for idx, seg in enumerate(audio_segments):
                if progress_callback:
                    progress_callback(0.3 + (0.6 * idx / total_segments), f"Processing segment {idx + 1}/{total_segments}...")

                start_seconds = parse_timestamp_to_seconds(seg["start"])
                end_seconds = parse_timestamp_to_seconds(seg["end"])
                time_str = format_seconds_to_timestamp(start_seconds)

                aligned_content = f"""## Segment {idx + 1} [{time_str}]

### Speaker Audio
{seg["text"]}
"""

                embedding = generate_embedding(aligned_content)

                supabase.from_("talk_chunks").insert({
                    "talk_id": talk_id,
                    "content": aligned_content,
                    "content_type": "aligned_segment",
                    "source_file": audio_file.name,
                    "slide_number": None,
                    "chunk_index": idx,
                    "start_time_seconds": start_seconds,
                    "end_time_seconds": end_seconds,
                    "embedding": embedding
                }).execute()

                results["segments"] += 1

        # Case 3: Slides only - create segments from slides (use EXIF if available)
        elif slide_files and not audio_file:
            total_slides = len(slides_with_time)
            for idx, slide in enumerate(slides_with_time):
                if progress_callback:
                    progress_callback(0.3 + (0.6 * idx / total_slides), f"Processing slide {idx + 1}/{total_slides}...")

                img = Image.open(io.BytesIO(slide["bytes"]))
                ocr_text = extract_slide_ocr(img)
                vision_desc = describe_slide_vision(img)
                thumbnail = create_thumbnail_base64(slide["bytes"])

                time_str = format_seconds_to_timestamp(slide.get("relative_seconds"))
                next_slide_time = slides_with_time[idx + 1].get("relative_seconds") if idx + 1 < len(slides_with_time) else None

                aligned_content = f"""## Slide {idx + 1} [{time_str}]

### Slide Text
{ocr_text if ocr_text else "[No text detected]"}

### Visual Description
{vision_desc}
"""

                embedding = generate_embedding(aligned_content)

                supabase.from_("talk_chunks").insert({
                    "talk_id": talk_id,
                    "content": aligned_content,
                    "content_type": "aligned_segment",
                    "source_file": slide["name"],
                    "slide_number": idx + 1,
                    "chunk_index": 0,
                    "start_time_seconds": slide.get("relative_seconds"),
                    "end_time_seconds": next_slide_time,
                    "embedding": embedding,
                    "slide_thumbnail": thumbnail
                }).execute()

                results["segments"] += 1

        if progress_callback:
            progress_callback(1.0, "Processing complete!")

        results["messages"].append(f"Created {results['segments']} aligned segments")

    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"Error: {str(e)}")

    return results

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

    # Check for aligned segments first
    aligned_chunks = [c for c in chunks if c['content_type'] == 'aligned_segment']

    if aligned_chunks:
        aligned_content = "\n\n".join([
            f"**[{format_seconds_to_timestamp(c.get('start_time_seconds'))}] Slide {c['slide_number']}:**\n{c['content']}"
            for c in sorted(aligned_chunks, key=lambda x: x.get('start_time_seconds') or 0)
        ])
        content_text = aligned_content[:12000]
    else:
        audio_text = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
        slides_text = "\n".join([c['content'] for c in chunks if c['content_type'] == 'slide_ocr'])
        content_text = f"{audio_text[:8000]}\n\n{slides_text[:3000]}"

    prompt = f"""Extract the most memorable and impactful quotes, statistics, and key statements from this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

**Content:**
{content_text}

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

    # Check for aligned segments first
    aligned_chunks = [c for c in chunks if c['content_type'] == 'aligned_segment']

    if aligned_chunks:
        aligned_content = "\n\n".join([
            f"**[{format_seconds_to_timestamp(c.get('start_time_seconds'))}] Slide {c['slide_number']}:**\n{c['content']}"
            for c in sorted(aligned_chunks, key=lambda x: x.get('start_time_seconds') or 0)
        ])
        content_text = aligned_content[:12000]
    else:
        audio_text = " ".join([c['content'] for c in chunks if c['content_type'] == 'audio_transcript'])
        slides_text = "\n".join([c['content'] for c in chunks if c['content_type'] == 'slide_ocr'])
        content_text = f"{audio_text[:8000]}\n\n{slides_text[:3000]}"

    prompt = f"""Extract all action items, recommendations, and next steps from this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

**Content:**
{content_text}

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

    # Check for aligned segments first (preferred format)
    aligned_chunks = [c for c in chunks if c['content_type'] == 'aligned_segment']

    if aligned_chunks:
        # Use aligned content with timestamps
        aligned_content = "\n\n".join([
            f"**[{format_seconds_to_timestamp(c.get('start_time_seconds'))}] Slide {c['slide_number']}:**\n{c['content']}"
            for c in sorted(aligned_chunks, key=lambda x: x.get('start_time_seconds') or 0)
        ])

        prompt = f"""Create a comprehensive summary of this AWS re:Invent talk.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

The content below is organized by slide with timestamps, showing what was on screen and what the speaker said at each moment:

{aligned_content[:12000]}

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
    else:
        # Fallback to legacy separate audio/slides format
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


# ============== Multimodal Summary & Insights ==============

def fetch_from_storage(url: str) -> bytes:
    """Fetch file bytes from Supabase Storage URL."""
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def upload_audio_to_gemini_for_summary(audio_bytes: bytes, audio_name: str):
    """Upload audio to Gemini and wait for processing."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        uploaded = ai.files.upload(file=tmp_path)

        # Wait for processing
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = ai.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            raise Exception("Audio processing failed")

        return uploaded
    finally:
        os.unlink(tmp_path)


def generate_talk_summary_multimodal(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    """Generate summary using audio + slides via Gemini multimodal API.

    If multimodal content (audio_url + slide_urls) is available, sends everything
    to Gemini for analysis. Falls back to text-only if not available.
    """
    talk = get_talk_by_id(talk_id)

    # Check for multimodal content
    audio_url = talk.get('audio_url')
    slide_urls = talk.get('slide_urls', [])

    if not audio_url or not slide_urls:
        # Fall back to text-only summary
        return generate_talk_summary(talk_id, model, custom_instructions)

    try:
        # Fetch audio from storage
        audio_bytes = fetch_from_storage(audio_url)

        # Upload audio to Gemini
        gemini_audio = upload_audio_to_gemini_for_summary(audio_bytes, f"{talk_id}_summary.mp3")

        # Fetch slide images
        slide_images = [fetch_from_storage(url) for url in slide_urls]

        # Build prompt
        prompt = f"""You are analyzing a complete conference presentation. You have:
1. The FULL audio recording of the speaker
2. Photos of ALL slides shown during the talk

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}
**Event:** {talk.get('event', 'Unknown')}

Listen to the ENTIRE audio recording and look at ALL slides to create a comprehensive summary.

**Generate the following sections:**

## Summary
A comprehensive 2-3 paragraph overview of the entire talk

## Key Topics
- Bullet points of main topics covered

## AWS Services Mentioned
- List ALL AWS services discussed with brief context

## Key Takeaways
- Main learnings and insights from the talk

## Recommended Actions
- Actionable steps based on the talk content

Be technical and specific. Reference actual content from the slides and speaker.
Use markdown formatting.{f'''

**Additional user instructions:** {custom_instructions}''' if custom_instructions else ''}"""

        # Build multimodal contents
        contents = [gemini_audio]
        for i, img_bytes in enumerate(slide_images):
            contents.append(f"[Slide {i+1}]")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        contents.append(prompt)

        # Execute multimodal request
        gemini_model = get_gemini_model_for_multimodal(model)
        response = ai.models.generate_content(
            model=gemini_model,
            contents=contents
        )

        result = response.text

        # Save to database
        save_ai_content(talk_id, "summary", result, gemini_model)

        return result

    except Exception as e:
        # Fall back to text-only on error
        st.warning(f"Multimodal summary failed ({str(e)}), using text-only.")
        return generate_talk_summary(talk_id, model, custom_instructions)


def extract_key_quotes_multimodal(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    """Extract key quotes using audio + slides via Gemini multimodal API."""
    talk = get_talk_by_id(talk_id)

    audio_url = talk.get('audio_url')
    slide_urls = talk.get('slide_urls', [])

    if not audio_url or not slide_urls:
        return extract_key_quotes(talk_id, model, custom_instructions)

    try:
        audio_bytes = fetch_from_storage(audio_url)
        gemini_audio = upload_audio_to_gemini_for_summary(audio_bytes, f"{talk_id}_quotes.mp3")
        slide_images = [fetch_from_storage(url) for url in slide_urls]

        prompt = f"""You are analyzing a conference presentation to extract memorable quotes.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

Listen to the ENTIRE audio and look at ALL slides. Extract:

## Memorable Quotes
Find 5-10 of the most impactful, quotable statements from the speaker.
- Include the exact quote (or very close paraphrase)
- Add brief context for each
- Focus on insights that would be valuable to share

## Notable Statistics
Any specific numbers, percentages, or data points mentioned

## Key Insights
Unique perspectives or surprising revelations from the talk

Format each quote clearly with quotation marks and attribution.{f'''

**Additional instructions:** {custom_instructions}''' if custom_instructions else ''}"""

        contents = [gemini_audio]
        for i, img_bytes in enumerate(slide_images):
            contents.append(f"[Slide {i+1}]")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        contents.append(prompt)

        gemini_model = get_gemini_model_for_multimodal(model)
        response = ai.models.generate_content(model=gemini_model, contents=contents)
        result = response.text

        save_ai_content(talk_id, "quotes", result, gemini_model)
        return result

    except Exception as e:
        st.warning(f"Multimodal quotes failed ({str(e)}), using text-only.")
        return extract_key_quotes(talk_id, model, custom_instructions)


def extract_action_items_multimodal(talk_id: str, model: str = "gemini-2.5-flash", custom_instructions: str = None) -> str:
    """Extract action items using audio + slides via Gemini multimodal API."""
    talk = get_talk_by_id(talk_id)

    audio_url = talk.get('audio_url')
    slide_urls = talk.get('slide_urls', [])

    if not audio_url or not slide_urls:
        return extract_action_items(talk_id, model, custom_instructions)

    try:
        audio_bytes = fetch_from_storage(audio_url)
        gemini_audio = upload_audio_to_gemini_for_summary(audio_bytes, f"{talk_id}_actions.mp3")
        slide_images = [fetch_from_storage(url) for url in slide_urls]

        prompt = f"""You are analyzing a conference presentation to extract actionable recommendations.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}

Listen to the ENTIRE audio and look at ALL slides. Extract:

## Immediate Actions
Things to do right away based on the talk

## Technologies to Explore
AWS services, tools, or frameworks recommended

## Best Practices
Approaches and patterns suggested by the speaker

## Resources & Links
Any URLs, documentation, or resources mentioned

## Learning Path
Suggested next steps for deeper learning

Be specific and actionable. Include context for why each item matters.{f'''

**Additional instructions:** {custom_instructions}''' if custom_instructions else ''}"""

        contents = [gemini_audio]
        for i, img_bytes in enumerate(slide_images):
            contents.append(f"[Slide {i+1}]")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        contents.append(prompt)

        gemini_model = get_gemini_model_for_multimodal(model)
        response = ai.models.generate_content(model=gemini_model, contents=contents)
        result = response.text

        save_ai_content(talk_id, "actions", result, gemini_model)
        return result

    except Exception as e:
        st.warning(f"Multimodal actions failed ({str(e)}), using text-only.")
        return extract_action_items(talk_id, model, custom_instructions)


def export_to_markdown(talk_id: str, include_transcript: bool = True) -> str:
    talk = get_talk_by_id(talk_id)
    chunks = get_talk_chunks(talk_id)
    summary = generate_talk_summary(talk_id)

    md = f"""# {talk['title']}

**Speaker:** {talk.get('speaker', 'Unknown')}
**Event:** {talk.get('event', DEFAULT_EVENT)}

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
if "chat_talk_id" not in st.session_state:
    st.session_state.chat_talk_id = None
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

# Get talks
talks = get_all_talks()

# Get available models (based on configured API keys)
available_models = get_available_models()

# ============== Top Navigation ==============

st.markdown("#### Conference Talk Notes")
st.caption(DEFAULT_EVENT)
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
                    # Thumbnail preview
                    if t.get('first_thumbnail'):
                        st.image(f"data:image/jpeg;base64,{t['first_thumbnail']}", use_container_width=True)

                    # Title
                    st.markdown(f"#### {t['title']}")

                    # Event badge
                    event_name = t.get('event') or DEFAULT_EVENT
                    st.caption(f":material/event: {event_name}")

                    # Speaker
                    if t.get('speaker'):
                        st.caption(f":material/person: {t['speaker']}")

                    # Stats row
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption(f":material/segment: {t.get('segment_count', 0)} segments")
                    with col_b:
                        # Status indicator
                        if t.get('has_summary'):
                            st.caption(":material/check_circle: Processed")
                        elif t.get('segment_count', 0) > 0:
                            st.caption(":material/pending: Uploaded")
                        else:
                            st.caption(":material/hourglass_empty: Empty")

                    # Created date
                    created = t.get('created_at', '')[:10]
                    st.caption(f":material/calendar_today: {created}")

                    # Action buttons
                    col_open, col_del = st.columns([3, 1])
                    with col_open:
                        if st.button("Open", key=f"open_{t['id']}", use_container_width=True, icon=":material/arrow_forward:"):
                            st.session_state.selected_talk = t["id"]
                            st.session_state.active_view = "talk_detail"
                            st.rerun()
                    with col_del:
                        if st.button("", key=f"del_{t['id']}", icon=":material/delete:", help="Delete talk"):
                            delete_talk_dialog(t["id"], t["title"])

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
    # Count segments
    aligned_segments = [c for c in chunks if c.get('content_type') == 'aligned_segment']
    legacy_chunks = [c for c in chunks if c.get('content_type') in ('audio_transcript', 'slide_ocr', 'slide_vision')]
    segment_count = len(aligned_segments) or len(legacy_chunks)
    total_chars = sum(len(c['content']) for c in chunks)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Segments", segment_count)
    with col2:
        st.metric("Total Characters", f"{total_chars:,}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([":material/upload: Upload", ":material/play_circle: Replay", ":material/summarize: Summary", ":material/lightbulb: Insights", ":material/search: Search", ":material/download: Export", ":material/settings: Manage"])

    with tab1:
        # ========== PROCESSING STATE MACHINE ==========
        processing_step = st.session_state.get('processing_step', 'idle')

        # Check if we're processing a different talk
        if st.session_state.get('pending_talk_id') and st.session_state.pending_talk_id != talk['id']:
            reset_processing_state()
            processing_step = 'idle'

        # ========== STEP: IDLE (Upload Form) ==========
        if processing_step == 'idle':
            st.markdown("### Upload Content")
            st.caption("Upload audio and slides for step-by-step processing with validation.")

            col1, col2 = st.columns(2)
            with col1:
                upload_audio = st.file_uploader(
                    "Audio Recording",
                    type=["mp3", "mp4", "m4a", "wav", "webm"],
                    key=f"upload_audio_{st.session_state.upload_counter}"
                )
            with col2:
                upload_slides = st.file_uploader(
                    "Slide Photos",
                    type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                    accept_multiple_files=True,
                    key=f"upload_slides_{st.session_state.upload_counter}",
                    help="Photos with EXIF timestamps will be sorted and aligned"
                )

            # Show what will be processed
            if upload_audio or upload_slides:
                status_parts = []
                if upload_audio:
                    status_parts.append(f"1 audio file")
                if upload_slides:
                    status_parts.append(f"{len(upload_slides)} slides")
                st.markdown(f"**Ready to process:** {' + '.join(status_parts)}")

                if st.button("Start Processing", type="primary", use_container_width=True, icon=":material/auto_awesome:"):
                    # Store files in session state
                    st.session_state.pending_talk_id = talk['id']
                    if upload_audio:
                        st.session_state.pending_audio_file = {
                            'bytes': upload_audio.getvalue(),
                            'name': upload_audio.name
                        }
                    else:
                        st.session_state.pending_audio_file = None

                    if upload_slides:
                        st.session_state.pending_slide_files = [
                            {'bytes': f.getvalue(), 'name': f.name} for f in upload_slides
                        ]
                    else:
                        st.session_state.pending_slide_files = []

                    # Determine next step based on what was uploaded
                    if upload_audio and upload_slides:
                        # UNIFIED MULTIMODAL: Both audio and slides
                        st.session_state.processing_step = 'multimodal_processing'
                    elif upload_audio:
                        # Audio only - use old transcription flow
                        st.session_state.processing_step = 'transcribing'
                    elif upload_slides:
                        # Slides only
                        st.session_state.processing_step = 'preparing_slides_only'
                    st.rerun()

            # Previously uploaded content section
            st.divider()
            st.markdown("### Processed Content")
            aligned_count = len([c for c in chunks if c.get('content_type') == 'aligned_segment'])
            if aligned_count > 0:
                st.caption(f":material/check_circle: {aligned_count} segments processed")
            else:
                st.caption("No content processed yet")

        # ========== STEP: TRANSCRIBING ==========
        elif processing_step == 'transcribing':
            st.markdown("### Step 1: Transcribing Audio")
            st.caption("Sending audio to Gemini for transcription...")

            with st.spinner("Transcribing audio... This may take a few minutes."):
                try:
                    result = step_transcribe_audio(
                        st.session_state.pending_audio_file['bytes'],
                        st.session_state.pending_audio_file['name']
                    )
                    st.session_state.transcript_segments = result['segments']
                    st.session_state.transcript_messages = result['messages']
                    st.session_state.processing_step = 'review_transcript'
                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: REVIEW TRANSCRIPT ==========
        elif processing_step == 'review_transcript':
            st.markdown("### Step 1: Review Transcription")

            segments = st.session_state.transcript_segments
            messages = st.session_state.get('transcript_messages', [])

            # Show status messages
            for msg in messages:
                st.info(msg)

            st.success(f"Gemini returned **{len(segments)} segments**")

            # Summary stats
            if segments:
                total_duration = parse_timestamp_to_seconds(segments[-1]['end'])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Segments", len(segments))
                with col2:
                    st.metric("Duration", format_seconds_to_timestamp(total_duration))

            # Scrollable transcript table
            st.markdown("#### Transcript Segments")
            with st.container(height=400):
                for i, seg in enumerate(segments):
                    cols = st.columns([1, 1, 8])
                    cols[0].code(seg['start'], language=None)
                    cols[1].code(seg['end'], language=None)
                    text_preview = seg['text'][:150] + '...' if len(seg['text']) > 150 else seg['text']
                    cols[2].text(text_preview)

            # Raw JSON expander
            with st.expander("Raw JSON Response"):
                st.json(segments)

            # Action buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve & Continue", type="primary", use_container_width=True, icon=":material/check:"):
                    if st.session_state.pending_slide_files:
                        st.session_state.processing_step = 'preparing_slides'
                    else:
                        # Audio only - store segments directly
                        st.session_state.processing_step = 'storing_audio_only'
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: PREPARING SLIDES ==========
        elif processing_step == 'preparing_slides':
            st.markdown("### Step 2: Processing Slides")
            st.caption("Extracting EXIF timestamps and uploading audio...")

            with st.spinner("Processing slides and audio metadata..."):
                try:
                    # Prepare slides
                    slides_result = step_prepare_slides(st.session_state.pending_slide_files)
                    st.session_state.slides_with_time = slides_result['slides_with_time']
                    st.session_state.use_exif = slides_result['use_exif_alignment']
                    st.session_state.slides_messages = slides_result['messages']

                    # Upload audio to storage
                    audio_result = step_upload_audio(
                        st.session_state.pending_audio_file['bytes'],
                        st.session_state.pending_audio_file['name'],
                        st.session_state.pending_talk_id
                    )
                    st.session_state.audio_url = audio_result['audio_url']
                    st.session_state.audio_start_time = audio_result['audio_start_time']
                    st.session_state.audio_messages = audio_result['messages']

                    st.session_state.processing_step = 'aligning'
                    st.rerun()
                except Exception as e:
                    st.error(f"Slide processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: ALIGNING ==========
        elif processing_step == 'aligning':
            st.markdown("### Step 2: Aligning Audio to Slides")

            # Show previous step messages
            for msg in st.session_state.get('slides_messages', []):
                st.info(msg)
            for msg in st.session_state.get('audio_messages', []):
                st.info(msg)

            with st.spinner("Running alignment..."):
                try:
                    result = step_align_slides_to_audio(
                        st.session_state.slides_with_time,
                        st.session_state.transcript_segments,
                        st.session_state.audio_start_time,
                        st.session_state.use_exif
                    )
                    st.session_state.alignment_result = result['alignment']
                    st.session_state.alignment_messages = result['messages']
                    st.session_state.processing_step = 'review_alignment'
                    st.rerun()
                except Exception as e:
                    st.error(f"Alignment failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: REVIEW ALIGNMENT ==========
        elif processing_step == 'review_alignment':
            st.markdown("### Step 2: Review Slide-Audio Alignment")

            alignment = st.session_state.alignment_result
            messages = st.session_state.get('alignment_messages', [])

            # Show status messages
            for msg in messages:
                st.info(msg)

            st.success(f"Aligned **{len(alignment)} slides** to audio")

            # Show each slide's alignment
            for item in alignment:
                start_ts = format_seconds_to_timestamp(item['start_time_seconds'])
                end_ts = format_seconds_to_timestamp(item['end_time_seconds'])
                with st.expander(f"Slide {item['slide_number']}: {start_ts} - {end_ts}"):
                    cols = st.columns([1, 2])

                    # Left: thumbnail
                    with cols[0]:
                        if item.get('thumbnail'):
                            st.image(f"data:image/jpeg;base64,{item['thumbnail']}", use_container_width=True)
                        else:
                            st.info("No thumbnail")

                    # Right: matched audio + OCR
                    with cols[1]:
                        st.markdown("**Matched Audio:**")
                        audio_text = item.get('matched_audio', '')
                        audio_preview = audio_text[:400] + '...' if len(audio_text) > 400 else audio_text
                        st.text(audio_preview)

                        if item.get('ocr_text'):
                            st.markdown("**OCR Text:**")
                            st.text(item['ocr_text'][:200])

            # Action buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve & Store", type="primary", use_container_width=True, icon=":material/save:"):
                    st.session_state.processing_step = 'storing'
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: STORING ==========
        elif processing_step == 'storing':
            st.markdown("### Step 3: Storing Data")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing aligned segments..."):
                try:
                    audio_name = st.session_state.pending_audio_file['name'] if st.session_state.pending_audio_file else ""
                    result = step_store_segments(
                        st.session_state.pending_talk_id,
                        st.session_state.alignment_result,
                        audio_name
                    )
                    st.session_state.store_messages = result['messages']
                    st.session_state.final_segment_count = result['segment_count']
                    st.session_state.processing_step = 'complete'
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: STORING AUDIO ONLY ==========
        elif processing_step == 'storing_audio_only':
            st.markdown("### Step 2: Storing Audio Segments")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing audio segments..."):
                try:
                    segments = st.session_state.transcript_segments
                    audio_name = st.session_state.pending_audio_file['name']
                    stored = 0

                    for idx, seg in enumerate(segments):
                        start_seconds = parse_timestamp_to_seconds(seg["start"])
                        end_seconds = parse_timestamp_to_seconds(seg["end"])
                        time_str = format_seconds_to_timestamp(start_seconds)

                        aligned_content = f"""## Segment {idx + 1} [{time_str}]

### Speaker Audio
{seg["text"]}
"""
                        embedding = generate_embedding(aligned_content)

                        supabase.from_("talk_chunks").insert({
                            "talk_id": st.session_state.pending_talk_id,
                            "content": aligned_content,
                            "content_type": "aligned_segment",
                            "source_file": audio_name,
                            "slide_number": None,
                            "chunk_index": idx,
                            "start_time_seconds": start_seconds,
                            "end_time_seconds": end_seconds,
                            "embedding": embedding
                        }).execute()
                        stored += 1

                    st.session_state.final_segment_count = stored
                    st.session_state.processing_step = 'complete'
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: MULTIMODAL PROCESSING ==========
        elif processing_step == 'multimodal_processing':
            st.markdown("### Unified Audio + Slides Processing")

            # Determine which model will be used
            selected = st.session_state.selected_model
            gemini_model = get_gemini_model_for_multimodal(selected)

            # Show info about model being used
            if not selected.startswith("gemini"):
                st.info(f"Multimodal processing requires Gemini. Using **{gemini_model}** (you selected {selected}).")
            else:
                st.caption(f"Using **{gemini_model}** for unified analysis...")

            # Show file info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Audio File", st.session_state.pending_audio_file['name'])
            with col2:
                st.metric("Slides", f"{len(st.session_state.pending_slide_files)} photos")

            with st.spinner(f"Processing with {gemini_model}... This may take a few minutes for longer talks."):
                try:
                    result = step_process_multimodal(
                        st.session_state.pending_audio_file,
                        st.session_state.pending_slide_files,
                        st.session_state.pending_talk_id,
                        model=selected  # Will be converted to Gemini model internally
                    )
                    st.session_state.multimodal_result = result
                    st.session_state.alignment_result = result['alignment']
                    st.session_state.multimodal_messages = result['messages']
                    st.session_state.multimodal_model_used = result.get('model_used', gemini_model)
                    st.session_state.processing_step = 'multimodal_review'
                    st.rerun()
                except Exception as e:
                    st.error(f"Multimodal processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: MULTIMODAL REVIEW ==========
        elif processing_step == 'multimodal_review':
            st.markdown("### Review Unified Analysis")

            alignment = st.session_state.alignment_result
            messages = st.session_state.get('multimodal_messages', [])
            model_used = st.session_state.get('multimodal_model_used', 'Gemini')

            # Show status messages
            for msg in messages:
                st.info(msg)

            st.success(f"**{model_used}** analyzed **{len(alignment)} slides** with audio")

            # Show each slide's analysis
            for item in alignment:
                start_ts = format_seconds_to_timestamp(item.get('start_time_seconds'))
                end_ts = format_seconds_to_timestamp(item.get('end_time_seconds'))
                with st.expander(f"Slide {item['slide_number']}: {start_ts} - {end_ts}", expanded=False):
                    cols = st.columns([1, 2])

                    # Left: thumbnail
                    with cols[0]:
                        if item.get('thumbnail'):
                            st.image(f"data:image/jpeg;base64,{item['thumbnail']}", use_container_width=True)
                        else:
                            st.info("No thumbnail")

                    # Right: content analysis
                    with cols[1]:
                        # OCR Text
                        ocr_text = item.get('ocr_text', '')
                        if ocr_text:
                            st.markdown("**Slide Text:**")
                            st.text(ocr_text[:300] + '...' if len(ocr_text) > 300 else ocr_text)

                        # Visual description
                        visual_desc = item.get('visual_description', '')
                        if visual_desc:
                            st.markdown("**Visual Elements:**")
                            st.text(visual_desc[:200] + '...' if len(visual_desc) > 200 else visual_desc)

                        # Speaker transcript
                        transcript = item.get('transcript_text', '')
                        if transcript:
                            st.markdown("**Speaker Said:**")
                            st.text(transcript[:400] + '...' if len(transcript) > 400 else transcript)

                        # Key points
                        key_points = item.get('key_points', [])
                        if key_points:
                            st.markdown("**Key Points:**")
                            for kp in key_points[:3]:
                                st.markdown(f"- {kp}")

            # Action buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve & Store", type="primary", use_container_width=True, icon=":material/save:"):
                    st.session_state.processing_step = 'multimodal_storing'
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: MULTIMODAL STORING ==========
        elif processing_step == 'multimodal_storing':
            st.markdown("### Storing Data")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing aligned segments..."):
                try:
                    audio_name = st.session_state.pending_audio_file['name']
                    result = step_store_multimodal_results(
                        st.session_state.pending_talk_id,
                        st.session_state.alignment_result,
                        audio_name
                    )
                    st.session_state.store_messages = result['messages']
                    st.session_state.final_segment_count = result['segment_count']
                    st.session_state.processing_step = 'complete'
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: PREPARING SLIDES ONLY ==========
        elif processing_step == 'preparing_slides_only':
            st.markdown("### Processing Slides")
            st.caption("Extracting EXIF timestamps and processing slides...")

            with st.spinner("Processing slides..."):
                try:
                    slides_result = step_prepare_slides(st.session_state.pending_slide_files)
                    slides_with_time = slides_result['slides_with_time']

                    stored = 0
                    for idx, slide in enumerate(slides_with_time):
                        img = Image.open(io.BytesIO(slide["bytes"]))
                        ocr_text = extract_slide_ocr(img)
                        vision_desc = describe_slide_vision(img)
                        thumbnail = create_thumbnail_base64(slide["bytes"])

                        time_str = format_seconds_to_timestamp(slide.get("relative_seconds"))
                        next_slide_time = slides_with_time[idx + 1].get("relative_seconds") if idx + 1 < len(slides_with_time) else None

                        aligned_content = f"""## Slide {idx + 1} [{time_str}]

### Slide Text
{ocr_text if ocr_text else "[No text detected]"}

### Visual Description
{vision_desc}
"""
                        embedding = generate_embedding(aligned_content)

                        supabase.from_("talk_chunks").insert({
                            "talk_id": st.session_state.pending_talk_id,
                            "content": aligned_content,
                            "content_type": "aligned_segment",
                            "source_file": slide["name"],
                            "slide_number": idx + 1,
                            "chunk_index": 0,
                            "start_time_seconds": slide.get("relative_seconds"),
                            "end_time_seconds": next_slide_time,
                            "embedding": embedding,
                            "slide_thumbnail": thumbnail
                        }).execute()
                        stored += 1

                    st.session_state.final_segment_count = stored
                    st.session_state.processing_step = 'complete'
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: COMPLETE ==========
        elif processing_step == 'complete':
            st.markdown("### Processing Complete!")

            segment_count = st.session_state.get('final_segment_count', 0)
            st.success(f"Successfully processed **{segment_count} segments**")

            # Only show balloons once (not on every rerun)
            if not st.session_state.get('balloons_shown', False):
                st.balloons()
                st.session_state.balloons_shown = True

            # Show any final messages
            for msg in st.session_state.get('store_messages', []):
                st.info(msg)

            if st.button("Process More Content", type="primary", use_container_width=True):
                reset_processing_state()
                st.session_state.upload_counter += 1
                st.rerun()

    with tab2:
        st.markdown("### Talk Replay")

        # Get aligned segments
        aligned_chunks = [c for c in chunks if c.get('content_type') == 'aligned_segment']

        if not aligned_chunks:
            # Check for legacy content
            if chunks:
                st.info("This talk was processed without timestamp alignment. Re-upload using 'Aligned (Audio + Slides)' for timeline view.")
            else:
                st.info("Upload content using 'Aligned (Audio + Slides)' to see the replay.")
        else:
            # Sort by timestamp
            sorted_chunks = sorted(aligned_chunks, key=lambda x: x.get('start_time_seconds') or 0)

            # Initialize navigation state (reset when switching talks)
            if 'replay_talk_id' not in st.session_state or st.session_state.replay_talk_id != talk['id']:
                st.session_state.replay_talk_id = talk['id']
                st.session_state.replay_slide_idx = 0

            # Clamp index to valid range
            current_idx = min(st.session_state.replay_slide_idx, len(sorted_chunks) - 1)
            current_chunk = sorted_chunks[current_idx]

            # Summary stats
            last_chunk = sorted_chunks[-1]
            total_duration = last_chunk.get('end_time_seconds') or last_chunk.get('start_time_seconds') or 0
            st.caption(f"Duration: {format_seconds_to_timestamp(total_duration)} | {len(sorted_chunks)} slides")

            # Two-column layout: Left = Slide + Controls, Right = Full Transcript
            col_left, col_right = st.columns([6, 4])

            with col_left:
                # Current slide image (large)
                thumbnail = current_chunk.get('slide_thumbnail')
                if thumbnail:
                    st.image(f"data:image/jpeg;base64,{thumbnail}", use_container_width=True)
                else:
                    st.info("No image for this slide")

                # Navigation row
                nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                with nav_col1:
                    if st.button(":material/chevron_left:", disabled=(current_idx == 0), key="nav_prev", use_container_width=True):
                        st.session_state.replay_slide_idx = current_idx - 1
                        st.rerun()
                with nav_col2:
                    start_time = current_chunk.get('start_time_seconds')
                    timestamp_str = format_seconds_to_timestamp(start_time) if start_time is not None else "??:??"
                    st.markdown(f"**Slide {current_idx + 1} of {len(sorted_chunks)}** `{timestamp_str}`")
                with nav_col3:
                    if st.button(":material/chevron_right:", disabled=(current_idx >= len(sorted_chunks) - 1), key="nav_next", use_container_width=True):
                        st.session_state.replay_slide_idx = current_idx + 1
                        st.rerun()

                # Audio player
                audio_url = talk.get('audio_url')
                if audio_url:
                    st.audio(audio_url)

                # Slide OCR/Visual for current slide (collapsible)
                current_sections = parse_aligned_content(current_chunk['content'])
                if current_sections.get('slide_text'):
                    with st.expander("Slide Text (OCR)", expanded=False, icon=":material/text_fields:"):
                        st.markdown(current_sections['slide_text'])
                if current_sections.get('visual_desc'):
                    with st.expander("Visual Notes", expanded=False, icon=":material/visibility:"):
                        st.markdown(current_sections['visual_desc'])

            with col_right:
                st.markdown("#### Full Transcript")

                # Scrollable container with ALL transcripts
                with st.container(height=500):
                    for idx, chunk in enumerate(sorted_chunks):
                        sections = parse_aligned_content(chunk['content'])
                        slide_num = chunk.get('slide_number', idx + 1)
                        is_current = (idx == current_idx)
                        start_time = chunk.get('start_time_seconds')
                        ts_str = format_seconds_to_timestamp(start_time) if start_time is not None else "??:??"

                        # Slide header with visual distinction for current slide
                        if is_current:
                            st.markdown(f"**:violet[▶ Slide {slide_num}]** `{ts_str}`")
                        else:
                            # Clickable slide header to navigate
                            if st.button(f"Slide {slide_num} `{ts_str}`", key=f"slide_btn_{idx}", use_container_width=True):
                                st.session_state.replay_slide_idx = idx
                                st.rerun()

                        # Transcript content
                        if sections.get('audio') and sections['audio'] != '_No matching audio_':
                            transcript_text = sections['audio'].strip()
                            if is_current:
                                # Full transcript for current slide
                                st.markdown(transcript_text)
                            else:
                                # Truncated preview for other slides
                                preview = transcript_text[:150] + "..." if len(transcript_text) > 150 else transcript_text
                                st.caption(preview)
                        else:
                            st.caption("_No transcript_")

                        st.markdown("---")

            # Thumbnail strip at bottom for quick navigation
            st.markdown("##### Quick Jump")
            # Show up to 15 thumbnails, or all if fewer
            max_thumbs = min(len(sorted_chunks), 15)
            thumb_cols = st.columns(max_thumbs)
            for i in range(max_thumbs):
                with thumb_cols[i]:
                    is_selected = (i == current_idx)
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(f"{i+1}", key=f"thumb_{i}", type=btn_type, use_container_width=True):
                        st.session_state.replay_slide_idx = i
                        st.rerun()

    with tab3:
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
                with st.spinner("Generating comprehensive summary (multimodal when available)..."):
                    generate_talk_summary_multimodal(talk["id"], st.session_state.selected_model, summary_instructions or None)
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

    with tab4:
        st.markdown("### Talk Insights")

        if not chunks:
            st.info("Upload audio or slides first to generate insights.")
        else:
            # Load saved chat history (reset if switching talks)
            if st.session_state.chat_talk_id != talk["id"]:
                st.session_state.chat_talk_id = talk["id"]
                saved_chat = get_chat_history(talk["id"])
                st.session_state.chat_history = saved_chat if saved_chat else []

            # Key Quotes Section
            st.markdown("#### Key Quotes & Highlights")
            quotes_instructions = st.text_area(
                "Custom instructions (optional)",
                placeholder="e.g., Focus on quotes about cost optimization...",
                key="quotes_instructions",
                height=68
            )
            if st.button("Extract Quotes", use_container_width=True, icon=":material/format_quote:"):
                with st.spinner("Extracting key quotes (multimodal when available)..."):
                    extract_key_quotes_multimodal(talk["id"], st.session_state.selected_model, quotes_instructions or None)
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
                with st.spinner("Extracting action items (multimodal when available)..."):
                    extract_action_items_multimodal(talk["id"], st.session_state.selected_model, actions_instructions or None)
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

    with tab5:
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

    with tab6:
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

    with tab7:
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

        if st.button("Delete Talk", type="secondary", icon=":material/delete_forever:"):
            delete_talk_dialog(talk["id"], talk["title"])

        st.markdown('</div>', unsafe_allow_html=True)
