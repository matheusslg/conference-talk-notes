import streamlit as st

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Conference Talk Notes",
    layout="wide"
)

from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif
import io
import json
import base64
import re

# Import from src modules
from src.config import (
    DEFAULT_EVENT,
    DEFAULT_MODEL,
    MAX_AUDIO_SIZE_BYTES,
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    UI_THUMBNAIL_WIDTH,
    UI_TRANSCRIPT_HEIGHT,
    UI_FULL_TRANSCRIPT_HEIGHT,
    UI_TEXT_AREA_HEIGHT,
    UI_MAX_THUMBNAIL_STRIP,
    ProcessingStep,
    SessionKey,
)
from src.utils import parse_timestamp_to_seconds, format_seconds_to_timestamp
from src.auth import check_password, get_user_name, set_user_name
from src.database import (
    supabase,
    create_talk,
    get_all_talks,
    get_talk_by_id,
    update_talk,
    delete_talk,
    get_talk_chunks,
    get_all_ai_content,
    delete_ai_content,
    get_chat_history,
    save_chat_history,
    upload_audio,
)
from src.llm import generate_embedding, get_available_models
from src.processing import (
    get_gemini_model_for_multimodal,
    create_thumbnail_base64,
    extract_talk_info_from_slide,
    extract_slide_ocr,
    describe_slide_vision,
    step_transcribe_audio,
    step_transcribe_audio_multi,
    step_prepare_slides,
    step_upload_audio,
    step_align_slides_to_audio,
    step_store_segments,
    step_process_multimodal,
    step_store_multimodal_results,
    ingest_audio,
    ingest_images,
    parse_aligned_content,
)
from src.search import search_talk_content, generate_search_insights
from src.insights import (
    generate_talk_summary,
    generate_talk_summary_multimodal,
    export_to_markdown,
    extract_key_quotes,
    extract_key_quotes_multimodal,
    extract_action_items_multimodal,
    chat_with_talk,
)

# Register HEIF/HEIC support with Pillow
pillow_heif.register_heif_opener()

# Gate the entire app
if not check_password():
    st.stop()

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

# ============== UI Dialogs ==============

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

def reset_processing_state():
    """Clear all processing-related session state."""
    keys_to_clear = [
        'processing_step', 'pending_talk_id', 'pending_audio_files',
        'pending_slide_files', 'transcript_segments', 'alignment_result',
        'slides_with_time', 'audio_start_time', 'audio_url', 'use_exif',
        'multimodal_result', 'multimodal_messages', 'multimodal_model_used',
        'balloons_shown', 'final_segment_count', 'store_messages'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def get_audio_display_name() -> str:
    """Get display name for audio files from session state.

    Returns first file name if single file, or combined name if multiple.
    """
    audio_files = st.session_state.get('pending_audio_files', [])
    if not audio_files:
        return ""
    if len(audio_files) == 1:
        return audio_files[0]['name']
    # Multiple files - show first name with count
    return f"{audio_files[0]['name']} (+{len(audio_files) - 1} more)"


def get_first_audio_file() -> dict | None:
    """Get the first audio file from session state, for backward compatibility."""
    audio_files = st.session_state.get('pending_audio_files', [])
    return audio_files[0] if audio_files else None


# ============== Processing ==============
# MAX_AUDIO_SIZE_BYTES is now imported from src.config

# Pipeline functions (process_talk_content, ingest_audio, ingest_images, etc.)
# have been moved to src/processing/pipeline.py and are imported at the top.


# Insights functions (extract_key_quotes, generate_talk_summary, chat_with_talk, etc.)
# have been moved to src/insights/ and are imported at the top.

# ============== Session State ==============

if "active_view" not in st.session_state:
    st.session_state.active_view = "talks"
if "selected_talk" not in st.session_state:
    st.session_state.selected_talk = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_talk_id" not in st.session_state:
    st.session_state.chat_talk_id = None
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0
if "current_user_name" not in st.session_state:
    st.session_state.current_user_name = None

# Get talks
talks = get_all_talks()

# Get available models (based on configured API keys)
available_models = get_available_models()

# ============== User Name Prompt ==============
# Ask for user name once per session (check cookie first, then database)
user_name = get_user_name()
if not user_name:
    # Try to get most recent author_name from database as default
    default_name = ""
    if talks and talks[0].get('author_name'):
        default_name = talks[0]['author_name']

    st.markdown("### Welcome to Conference Talk Notes")
    st.caption("Please enter your name to attribute your notes.")

    with st.form("user_name_form"):
        name_input = st.text_input("Your name", value=default_name, placeholder="e.g., John Doe")
        if st.form_submit_button("Continue", type="primary", use_container_width=True):
            if name_input.strip():
                set_user_name(name_input.strip())
                st.rerun()
            else:
                st.warning("Please enter your name")
    st.stop()

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
    st.caption(":material/auto_awesome: Upload a title slide to auto-fill title & speaker")

    # Image upload (optional) with preview
    col_upload, col_preview = st.columns([3, 1])
    with col_upload:
        title_slide = st.file_uploader(
            "Title slide (optional)",
            type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
            key="title_slide_uploader",
            label_visibility="collapsed"
        )
    with col_preview:
        if title_slide:
            img = Image.open(io.BytesIO(title_slide.getvalue()))
            st.image(img, width=UI_THUMBNAIL_WIDTH)

    # Auto-extract when image uploaded (track by filename)
    if title_slide:
        if st.session_state.get("last_title_slide") != title_slide.name:
            st.session_state.last_title_slide = title_slide.name
            with st.spinner("Extracting info..."):
                img = Image.open(io.BytesIO(title_slide.getvalue()))
                info = extract_talk_info_from_slide(img, st.session_state.selected_model)
                st.session_state.extracted_title = info.get("title") or ""
                st.session_state.extracted_speaker = info.get("speaker") or ""
            st.rerun()

    # Clear extraction state if no file
    if not title_slide and "last_title_slide" in st.session_state:
        del st.session_state.last_title_slide
        st.session_state.pop("extracted_title", None)
        st.session_state.pop("extracted_speaker", None)

    # Form with pre-populated values (from extraction or empty)
    with st.form("new_talk_form", clear_on_submit=True):
        default_title = st.session_state.get("extracted_title", "")
        default_speaker = st.session_state.get("extracted_speaker", "")

        new_title = st.text_input("Talk Title", value=default_title, placeholder="SVS301 - Building Serverless Apps...")
        new_speaker = st.text_input("Speaker (optional)", value=default_speaker, placeholder="John Doe")

        if st.form_submit_button("Create Talk", type="primary", use_container_width=True, icon=":material/add:"):
            if new_title:
                talk_id = create_talk(new_title, new_speaker or None, st.session_state.current_user_name)
                if talk_id:
                    # Clean up extraction state
                    for key in ["extracted_title", "extracted_speaker", "last_title_slide"]:
                        st.session_state.pop(key, None)
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

                    # Author (notes by)
                    if t.get('author_name'):
                        st.caption(f":material/edit: {t['author_name']}")

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
                insights = generate_search_insights(query, results, st.session_state.selected_model)

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
        st.session_state.selected_talk = None
        st.rerun()

    st.title(talk['title'])
    if talk.get('speaker'):
        st.caption(f"Speaker: {talk['speaker']}")
    if talk.get('author_name'):
        st.caption(f"Notes by: {talk['author_name']}")

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
        processing_step = st.session_state.get('processing_step', ProcessingStep.IDLE)

        # Check if we're processing a different talk
        if st.session_state.get('pending_talk_id') and st.session_state.pending_talk_id != talk['id']:
            reset_processing_state()
            processing_step = ProcessingStep.IDLE

        # ========== STEP: IDLE (Upload Form) ==========
        if processing_step == ProcessingStep.IDLE:
            st.markdown("### Upload Content")
            st.caption("Upload audio and slides together for unified processing.")

            # Unified file uploader - accepts both audio and images
            uploaded_files = st.file_uploader(
                "Audio & Slide Photos",
                type=AUDIO_EXTENSIONS + IMAGE_EXTENSIONS,
                accept_multiple_files=True,
                key=f"upload_files_{st.session_state.upload_counter}",
                help="Select audio recording and slide photos together"
            )

            # Separate files by type - now supports multiple audio files
            upload_audio_files = []
            upload_slides = []

            if uploaded_files:
                for f in uploaded_files:
                    ext = f.name.split('.')[-1].lower()
                    if ext in AUDIO_EXTENSIONS:
                        upload_audio_files.append(f)
                    elif ext in IMAGE_EXTENSIONS:
                        upload_slides.append(f)

            # Show what will be processed
            if upload_audio_files or upload_slides:
                col1, col2 = st.columns(2)
                with col1:
                    if upload_audio_files:
                        if len(upload_audio_files) == 1:
                            st.caption(f":material/audio_file: {upload_audio_files[0].name}")
                        else:
                            st.caption(f":material/audio_file: {len(upload_audio_files)} audio fragments")
                            for af in upload_audio_files:
                                st.caption(f"  • {af.name}")
                    else:
                        st.caption(":material/audio_file: No audio")
                with col2:
                    st.caption(f":material/image: {len(upload_slides)} slide(s)")

                if st.button("Start Processing", type="primary", use_container_width=True, icon=":material/auto_awesome:"):
                    # Store files in session state
                    st.session_state.pending_talk_id = talk['id']

                    # Store audio files as a list (supports multiple fragments)
                    if upload_audio_files:
                        st.session_state.pending_audio_files = [
                            {'bytes': f.getvalue(), 'name': f.name} for f in upload_audio_files
                        ]
                    else:
                        st.session_state.pending_audio_files = []

                    if upload_slides:
                        st.session_state.pending_slide_files = [
                            {'bytes': f.getvalue(), 'name': f.name} for f in upload_slides
                        ]
                    else:
                        st.session_state.pending_slide_files = []

                    # Determine next step based on what was uploaded
                    if upload_audio_files and upload_slides:
                        # UNIFIED MULTIMODAL: Both audio and slides
                        st.session_state.processing_step = ProcessingStep.MULTIMODAL_PROCESSING
                    elif upload_audio_files:
                        # Audio only - use transcription flow (supports multi-file)
                        st.session_state.processing_step = ProcessingStep.TRANSCRIBING
                    elif upload_slides:
                        # Slides only
                        st.session_state.processing_step = ProcessingStep.PREPARING_SLIDES_ONLY
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
        elif processing_step == ProcessingStep.TRANSCRIBING:
            audio_files = st.session_state.get('pending_audio_files', [])
            num_files = len(audio_files)

            if num_files > 1:
                st.markdown(f"### Step 1: Transcribing {num_files} Audio Fragments")
                st.caption("Processing multiple audio files and merging transcripts...")
            else:
                st.markdown("### Step 1: Transcribing Audio")
                st.caption("Sending audio to Gemini for transcription...")

            with st.spinner("Transcribing audio... This may take a few minutes."):
                try:
                    result = step_transcribe_audio_multi(
                        audio_files,
                        st.session_state.selected_model
                    )
                    st.session_state.transcript_segments = result['segments']
                    st.session_state.transcript_messages = result['messages']
                    st.session_state.transcript_audio_files = result.get('audio_files', [])
                    st.session_state.transcript_gaps = result.get('gaps', [])
                    st.session_state.processing_step = ProcessingStep.REVIEW_TRANSCRIPT
                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: REVIEW TRANSCRIPT ==========
        elif processing_step == ProcessingStep.REVIEW_TRANSCRIPT:
            st.markdown("### Step 1: Review Transcription")

            segments = st.session_state.transcript_segments
            messages = st.session_state.get('transcript_messages', [])
            audio_files = st.session_state.get('transcript_audio_files', [])
            gaps = st.session_state.get('transcript_gaps', [])

            # Show audio fragments info if multiple files
            if len(audio_files) > 1:
                with st.expander(f"Audio Fragments ({len(audio_files)} files)", expanded=True):
                    for i, af in enumerate(audio_files):
                        duration_str = format_seconds_to_timestamp(af['duration_seconds']) if af.get('duration_seconds') else "?"
                        time_str = af['creation_time'].strftime('%H:%M:%S') if af.get('creation_time') else "no timestamp"
                        st.caption(f"{i+1}. **{af['name']}** ({time_str}) - {duration_str}")
                        # Show gap after this file (if any)
                        for gap in gaps:
                            if gap.get('after_file') == af['name']:
                                gap_min = int(gap['gap_seconds'] // 60)
                                gap_sec = int(gap['gap_seconds'] % 60)
                                st.caption(f"   ↳ Gap: {gap_min}m {gap_sec}s")

            # Show status messages
            for msg in messages:
                st.info(msg)

            st.success(f"Gemini returned **{len(segments)} segments**")

            # Summary stats
            if segments:
                total_duration = parse_timestamp_to_seconds(segments[-1]['end'])
                if len(audio_files) > 1:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Segments", len(segments))
                    with col2:
                        st.metric("Duration", format_seconds_to_timestamp(total_duration))
                    with col3:
                        st.metric("Files", len(audio_files))
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Segments", len(segments))
                    with col2:
                        st.metric("Duration", format_seconds_to_timestamp(total_duration))

            # Scrollable transcript table
            st.markdown("#### Transcript Segments")
            with st.container(height=UI_TRANSCRIPT_HEIGHT):
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
                        st.session_state.processing_step = ProcessingStep.PREPARING_SLIDES
                    else:
                        # Audio only - store segments directly
                        st.session_state.processing_step = ProcessingStep.STORING_AUDIO_ONLY
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: PREPARING SLIDES ==========
        elif processing_step == ProcessingStep.PREPARING_SLIDES:
            st.markdown("### Step 2: Processing Slides")
            st.caption("Extracting EXIF timestamps and uploading audio...")

            with st.spinner("Processing slides and audio metadata..."):
                try:
                    # Prepare slides
                    slides_result = step_prepare_slides(st.session_state.pending_slide_files)
                    st.session_state.slides_with_time = slides_result['slides_with_time']
                    st.session_state.use_exif = slides_result['use_exif_alignment']
                    st.session_state.slides_messages = slides_result['messages']

                    # Upload audio to storage (use first file for traditional pipeline)
                    first_audio = get_first_audio_file()
                    audio_result = step_upload_audio(
                        first_audio['bytes'],
                        first_audio['name'],
                        st.session_state.pending_talk_id
                    )
                    st.session_state.audio_url = audio_result['audio_url']
                    st.session_state.audio_start_time = audio_result['audio_start_time']
                    st.session_state.audio_messages = audio_result['messages']

                    st.session_state.processing_step = ProcessingStep.ALIGNING
                    st.rerun()
                except Exception as e:
                    st.error(f"Slide processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: ALIGNING ==========
        elif processing_step == ProcessingStep.ALIGNING:
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
                        st.session_state.use_exif,
                        st.session_state.selected_model
                    )
                    st.session_state.alignment_result = result['alignment']
                    st.session_state.alignment_messages = result['messages']
                    st.session_state.processing_step = ProcessingStep.REVIEW_ALIGNMENT
                    st.rerun()
                except Exception as e:
                    st.error(f"Alignment failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: REVIEW ALIGNMENT ==========
        elif processing_step == ProcessingStep.REVIEW_ALIGNMENT:
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
                    st.session_state.processing_step = ProcessingStep.STORING
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: STORING ==========
        elif processing_step == ProcessingStep.STORING:
            st.markdown("### Step 3: Storing Data")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing aligned segments..."):
                try:
                    audio_name = get_audio_display_name()
                    result = step_store_segments(
                        st.session_state.pending_talk_id,
                        st.session_state.alignment_result,
                        audio_name
                    )
                    st.session_state.store_messages = result['messages']
                    st.session_state.final_segment_count = result['segment_count']
                    st.session_state.processing_step = ProcessingStep.COMPLETE
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: STORING AUDIO ONLY ==========
        elif processing_step == ProcessingStep.STORING_AUDIO_ONLY:
            st.markdown("### Step 2: Storing Audio Segments")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing audio segments..."):
                try:
                    segments = st.session_state.transcript_segments
                    audio_name = get_audio_display_name()
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
                    st.session_state.processing_step = ProcessingStep.COMPLETE
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: MULTIMODAL PROCESSING ==========
        elif processing_step == ProcessingStep.MULTIMODAL_PROCESSING:
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
                st.metric("Audio File", get_audio_display_name())
            with col2:
                st.metric("Slides", f"{len(st.session_state.pending_slide_files)} photos")

            with st.spinner(f"Processing with {gemini_model}... This may take a few minutes for longer talks."):
                try:
                    # For multimodal, use first audio file (multimodal doesn't support multiple yet)
                    first_audio = get_first_audio_file()
                    result = step_process_multimodal(
                        first_audio,
                        st.session_state.pending_slide_files,
                        st.session_state.pending_talk_id,
                        model=selected  # Will be converted to Gemini model internally
                    )
                    st.session_state.multimodal_result = result
                    st.session_state.alignment_result = result['alignment']
                    st.session_state.multimodal_messages = result['messages']
                    st.session_state.multimodal_model_used = result.get('model_used', gemini_model)
                    st.session_state.processing_step = ProcessingStep.MULTIMODAL_REVIEW
                    st.rerun()
                except Exception as e:
                    st.error(f"Multimodal processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: MULTIMODAL REVIEW ==========
        elif processing_step == ProcessingStep.MULTIMODAL_REVIEW:
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
                    st.session_state.processing_step = ProcessingStep.MULTIMODAL_STORING
                    st.rerun()
            with col2:
                if st.button("Abort Processing", type="secondary", use_container_width=True, icon=":material/close:"):
                    reset_processing_state()
                    st.session_state.upload_counter += 1
                    st.rerun()

        # ========== STEP: MULTIMODAL STORING ==========
        elif processing_step == ProcessingStep.MULTIMODAL_STORING:
            st.markdown("### Storing Data")
            st.caption("Generating embeddings and storing to database...")

            with st.spinner("Storing aligned segments..."):
                try:
                    audio_name = get_audio_display_name()
                    result = step_store_multimodal_results(
                        st.session_state.pending_talk_id,
                        st.session_state.alignment_result,
                        audio_name
                    )
                    st.session_state.store_messages = result['messages']
                    st.session_state.final_segment_count = result['segment_count']
                    st.session_state.processing_step = ProcessingStep.COMPLETE
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: PREPARING SLIDES ONLY ==========
        elif processing_step == ProcessingStep.PREPARING_SLIDES_ONLY:
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
                    st.session_state.processing_step = ProcessingStep.COMPLETE
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    if st.button("Abort", type="secondary"):
                        reset_processing_state()
                        st.rerun()

        # ========== STEP: COMPLETE ==========
        elif processing_step == ProcessingStep.COMPLETE:
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
                with st.container(height=UI_FULL_TRANSCRIPT_HEIGHT):
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
            max_thumbs = min(len(sorted_chunks), UI_MAX_THUMBNAIL_STRIP)
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
                height=UI_TEXT_AREA_HEIGHT
            )
            if st.button(f"Generate Summary ({st.session_state.selected_model})", type="primary", use_container_width=True, icon=":material/auto_awesome:"):
                with st.spinner("Generating comprehensive summary (multimodal when available)..."):
                    generate_talk_summary_multimodal(talk["id"], st.session_state.selected_model, summary_instructions or None, on_fallback_warning=st.warning)
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
                height=UI_TEXT_AREA_HEIGHT
            )
            if st.button("Extract Quotes", use_container_width=True, icon=":material/format_quote:"):
                with st.spinner("Extracting key quotes (multimodal when available)..."):
                    extract_key_quotes_multimodal(talk["id"], st.session_state.selected_model, quotes_instructions or None, on_fallback_warning=st.warning)
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
                height=UI_TEXT_AREA_HEIGHT
            )
            if st.button("Extract Actions", use_container_width=True, icon=":material/checklist:"):
                with st.spinner("Extracting action items (multimodal when available)..."):
                    extract_action_items_multimodal(talk["id"], st.session_state.selected_model, actions_instructions or None, on_fallback_warning=st.warning)
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
            edit_author = st.text_input("Notes by", value=talk.get("author_name") or "")

            if st.form_submit_button("Save Changes", type="primary", use_container_width=True, icon=":material/save:"):
                if edit_title:
                    update_talk(talk["id"], edit_title, edit_speaker or None, edit_author or None)
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
