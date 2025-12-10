"""Image processing - thumbnails, EXIF, OCR, vision."""

import io
import base64
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

from src.llm import ai
from src.processing.audio import get_gemini_model_for_multimodal


def create_thumbnail_base64(image_bytes: bytes, max_size: int = 800) -> str:
    """Create a readable thumbnail and return as base64 string."""
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


def extract_talk_info_from_slide(image: Image.Image, model: str = "gemini-2.5-pro") -> dict:
    """Extract talk title, speaker name, and session code from a title slide.

    Returns dict with keys: title, speaker, session_code (all may be None if not found).
    """
    # Convert to bytes for Gemini
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()

    from google.genai import types
    prompt = """Look at this conference slide image and extract:
1. The talk/session title
2. The speaker name(s)
3. The session code (like "SVS301", "SEC204", etc.)

Output ONLY a JSON object with these fields (use null if not found):
{"title": "...", "speaker": "...", "session_code": "..."}

Output valid JSON only, no markdown or explanation."""

    gemini_model = get_gemini_model_for_multimodal(model)
    response = ai.models.generate_content(
        model=gemini_model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )

    import json
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"title": None, "speaker": None, "session_code": None}


def extract_slide_ocr(image: Image.Image, model: str = "gemini-2.5-pro") -> str:
    """Extract all text from a slide image using OCR."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()

    from google.genai import types
    prompt = """Extract ALL text visible in this slide image.
Include: titles, bullet points, code, labels, footnotes.
Output the text exactly as it appears, preserving structure.
Output only the extracted text, no commentary."""

    gemini_model = get_gemini_model_for_multimodal(model)
    response = ai.models.generate_content(
        model=gemini_model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )

    return response.text.strip()


def describe_slide_vision(image: Image.Image, model: str = "gemini-2.5-pro") -> str:
    """Generate a description of slide content using vision."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()

    from google.genai import types
    prompt = """Describe this presentation slide:
1. What is the main topic or concept being presented?
2. What are the key points or takeaways?
3. Describe any diagrams, charts, or visual elements.
4. Note any code samples or technical details.

Be comprehensive but concise."""

    gemini_model = get_gemini_model_for_multimodal(model)
    response = ai.models.generate_content(
        model=gemini_model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )

    return response.text.strip()
