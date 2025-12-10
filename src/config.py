"""Configuration constants and environment settings."""

from datetime import datetime

# Event configuration
CURRENT_YEAR = datetime.now().year
DEFAULT_EVENT = f"AWS re:Invent {CURRENT_YEAR}"

# Supported file formats
SUPPORTED_AUDIO_FORMATS = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".qta"]
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"]

# MIME type mapping for audio files (for Gemini API)
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".qta": "audio/mp4",  # QTA is typically AAC/MP4 container
}

# Audio formats that need conversion to MP3 before Gemini upload
AUDIO_FORMATS_NEED_CONVERSION = {".qta"}

# Text chunking for embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Photo capture delay: typical delay between slide appearing and user taking photo
# User opens camera, waits for transitions, then snaps - usually 5-15 seconds
PHOTO_CAPTURE_DELAY_SECONDS = 10

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

# Default model for text generation
DEFAULT_MODEL = "gemini-2.5-pro"

# File upload limits
MAX_AUDIO_SIZE_BYTES = 20 * 1024 * 1024  # 20MB

# UI Dimensions
UI_THUMBNAIL_WIDTH = 80
UI_TRANSCRIPT_HEIGHT = 400
UI_FULL_TRANSCRIPT_HEIGHT = 500
UI_TEXT_AREA_HEIGHT = 68
UI_MAX_THUMBNAIL_STRIP = 15

# File extensions without dots (for Streamlit file_uploader)
AUDIO_EXTENSIONS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "qta"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "heic", "heif"]


class ProcessingStep:
    """Processing step names for the upload state machine."""
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    REVIEW_TRANSCRIPT = "review_transcript"
    PREPARING_SLIDES = "preparing_slides"
    ALIGNING = "aligning"
    REVIEW_ALIGNMENT = "review_alignment"
    STORING = "storing"
    STORING_AUDIO_ONLY = "storing_audio_only"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    MULTIMODAL_REVIEW = "multimodal_review"
    MULTIMODAL_STORING = "multimodal_storing"
    PREPARING_SLIDES_ONLY = "preparing_slides_only"
    COMPLETE = "complete"


class SessionKey:
    """Session state key names to avoid typos."""
    ACTIVE_VIEW = "active_view"
    SELECTED_TALK = "selected_talk"
    SELECTED_MODEL = "selected_model"
    CHAT_HISTORY = "chat_history"
    CHAT_TALK_ID = "chat_talk_id"
    UPLOAD_COUNTER = "upload_counter"
    CURRENT_USER_NAME = "current_user_name"
    PROCESSING_STEP = "processing_step"
    PROCESSING_TALK_ID = "processing_talk_id"
    TRANSCRIPT_RESULT = "transcript_result"
    SLIDES_RESULT = "slides_result"
    ALIGNMENT_RESULT = "alignment_result"
    AUDIO_FILE = "audio_file"
    SLIDE_FILES = "slide_files"
    MULTIMODAL_RESULT = "multimodal_result"


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
