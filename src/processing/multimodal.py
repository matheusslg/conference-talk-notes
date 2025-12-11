"""Multimodal processing functions for combined audio and slide analysis."""

import io

import json_repair
from PIL import Image
from google.genai import types

from src.llm import ai


# Supported image MIME types for Gemini
SUPPORTED_IMAGE_TYPES = {"jpeg", "png", "gif", "webp"}


def build_multimodal_prompt(slides_metadata: list) -> str:
    """Build the prompt for unified multimodal processing.

    Args:
        slides_metadata: List of slide metadata dicts with 'number', 'name',
                        and optionally 'relative_seconds'

    Returns:
        Formatted prompt string for Gemini multimodal request
    """
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


def normalize_image_for_gemini(img_bytes: bytes) -> tuple[bytes, str]:
    """Normalize image format for Gemini API compatibility.

    Args:
        img_bytes: Raw image bytes

    Returns:
        Tuple of (normalized_bytes, mime_type)
    """
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
        return buffer.getvalue(), "image/jpeg"

    mime_type = f"image/{fmt}"
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"

    return img_bytes, mime_type


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

    # Add each slide image with label
    for i, img_bytes in enumerate(slide_images):
        contents.append(f"[Slide {i+1}]")

        # Normalize image format
        normalized_bytes, mime_type = normalize_image_for_gemini(img_bytes)
        contents.append(types.Part.from_bytes(data=normalized_bytes, mime_type=mime_type))

    # Add the prompt
    prompt = build_multimodal_prompt(slides_metadata)
    contents.append(prompt)

    response = ai.models.generate_content(
        model=model,
        contents=contents
    )

    # Parse JSON response (use json_repair to handle malformed LLM output)
    text = response.text.strip()
    # Clean up markdown if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json_repair.loads(text)
