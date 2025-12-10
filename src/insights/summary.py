"""Summary generation and export functions."""

import os
import time
import tempfile

from google.genai import types

from src.config import DEFAULT_EVENT
from src.utils import format_seconds_to_timestamp, format_datetime_for_prompt
from src.database import get_talk_by_id, get_talk_chunks, save_ai_content
from src.llm import ai, generate_with_llm
from src.processing import build_slide_timing_context, get_gemini_model_for_multimodal


def fetch_from_storage(url: str) -> bytes:
    """Fetch file bytes from Supabase Storage URL."""
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def upload_audio_to_gemini_for_summary(audio_bytes: bytes, audio_name: str):
    """Upload audio to Gemini and wait for processing."""
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


def generate_talk_summary(talk_id: str, model: str = "gemini-2.5-pro", custom_instructions: str = None) -> str:
    """Generate a text-based summary of a talk."""
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


def generate_talk_summary_multimodal(
    talk_id: str,
    model: str = "gemini-2.5-pro",
    custom_instructions: str = None,
    on_fallback_warning=None
) -> str:
    """Generate summary using audio + slides via Gemini multimodal API.

    If multimodal content (audio_url + slide_urls) is available, sends everything
    to Gemini for analysis. Falls back to text-only if not available.

    Args:
        talk_id: Talk identifier
        model: Model to use
        custom_instructions: Optional additional instructions
        on_fallback_warning: Optional callback(message) when falling back to text-only
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

        # Normalize slide_urls to list of dicts (backwards compatibility)
        slide_metadata = []
        for slide_entry in slide_urls:
            if isinstance(slide_entry, dict):
                slide_metadata.append(slide_entry)
            else:
                # Old format: just a URL string
                slide_metadata.append({"url": slide_entry, "filename": None, "taken_at": None})

        # Fetch slide images
        slide_images = [fetch_from_storage(sm["url"]) for sm in slide_metadata]

        # Build slide timing context from aligned segments
        timing_context = build_slide_timing_context(talk_id, talk.get('audio_start_timestamp'))

        # Build prompt with timing information
        prompt = f"""You are analyzing a complete conference presentation. You have:
1. The FULL audio recording of the speaker
2. Photos of ALL slides shown during the talk
3. Precise timing information showing when each slide was discussed

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}
**Event:** {talk.get('event', 'Unknown')}
{f'''
{timing_context}
''' if timing_context else ''}
Use the slide timing information to understand the structure and flow of the talk.
Listen to the ENTIRE audio recording and look at ALL slides to create a comprehensive summary.

**Generate the following sections:**

## Summary
A comprehensive 2-3 paragraph overview of the entire talk

## Key Topics
- Bullet points of main topics covered (reference specific slides and timestamps when relevant)

## AWS Services Mentioned
- List ALL AWS services discussed with brief context

## Key Takeaways
- Main learnings and insights from the talk

## Recommended Actions
- Actionable steps based on the talk content

Be technical and specific. Reference actual content from the slides and speaker.
Use markdown formatting.{f'''

**Additional user instructions:** {custom_instructions}''' if custom_instructions else ''}"""

        # Build multimodal contents with chronological context
        contents = []

        # Add audio start timestamp context
        audio_start_ts = talk.get('audio_start_timestamp')
        if audio_start_ts:
            formatted_start = format_datetime_for_prompt(audio_start_ts)
            if formatted_start:
                contents.append(f"[Audio recording started at {formatted_start}]")

        # Add audio file
        contents.append(gemini_audio)

        # Add slides with photo metadata (filename, taken_at) for chronological context
        for i, img_bytes in enumerate(slide_images):
            slide_num = i + 1
            meta = slide_metadata[i]
            filename = meta.get("filename") or f"slide_{slide_num}"
            taken_at = meta.get("taken_at")

            if taken_at:
                formatted_taken = format_datetime_for_prompt(taken_at)
                contents.append(f'[Photo {slide_num} "{filename}" taken at {formatted_taken}]')
            else:
                contents.append(f'[Photo {slide_num} "{filename}"]')
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
        if on_fallback_warning:
            on_fallback_warning(f"Multimodal summary failed ({str(e)}), using text-only.")
        return generate_talk_summary(talk_id, model, custom_instructions)


def export_to_markdown(talk_id: str, include_transcript: bool = True) -> str:
    """Export talk to markdown format."""
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
