"""Action items extraction functions."""

from google.genai import types

from src.utils import format_seconds_to_timestamp, format_datetime_for_prompt
from src.database import get_talk_by_id, get_talk_chunks, save_ai_content
from src.llm import ai, generate_with_llm
from src.processing import build_slide_timing_context, get_gemini_model_for_multimodal
from src.insights.summary import fetch_from_storage, upload_audio_to_gemini_for_summary


def extract_action_items(talk_id: str, model: str = "gemini-2.5-pro", custom_instructions: str = None) -> str:
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


def extract_action_items_multimodal(
    talk_id: str,
    model: str = "gemini-2.5-pro",
    custom_instructions: str = None,
    on_fallback_warning=None
) -> str:
    """Extract action items using audio + slides via Gemini multimodal API.

    Args:
        talk_id: Talk identifier
        model: Model to use
        custom_instructions: Optional additional instructions
        on_fallback_warning: Optional callback(message) when falling back to text-only
    """
    talk = get_talk_by_id(talk_id)

    audio_url = talk.get('audio_url')
    slide_urls = talk.get('slide_urls', [])

    if not audio_url or not slide_urls:
        return extract_action_items(talk_id, model, custom_instructions)

    try:
        audio_bytes = fetch_from_storage(audio_url)
        gemini_audio = upload_audio_to_gemini_for_summary(audio_bytes, f"{talk_id}_actions.mp3")

        # Normalize slide_urls to list of dicts (backwards compatibility)
        slide_metadata = []
        for slide_entry in slide_urls:
            if isinstance(slide_entry, dict):
                slide_metadata.append(slide_entry)
            else:
                slide_metadata.append({"url": slide_entry, "filename": None, "taken_at": None})

        slide_images = [fetch_from_storage(sm["url"]) for sm in slide_metadata]

        # Build slide timing context
        timing_context = build_slide_timing_context(talk_id, talk.get('audio_start_timestamp'))

        prompt = f"""You are analyzing a conference presentation to extract actionable recommendations.

**Talk:** {talk['title']}
**Speaker:** {talk.get('speaker', 'Unknown')}
{f'''
{timing_context}
''' if timing_context else ''}
Listen to the ENTIRE audio and look at ALL slides. Extract:

## Immediate Actions
Things to do right away based on the talk (reference slide/timestamp where mentioned)

## Technologies to Explore
AWS services, tools, or frameworks recommended

## Best Practices
Approaches and patterns suggested by the speaker

## Resources & Links
Any URLs, documentation, or resources mentioned (with timestamp where they appear)

## Learning Path
Suggested next steps for deeper learning

Be specific and actionable. Include context for why each item matters.{f'''

**Additional instructions:** {custom_instructions}''' if custom_instructions else ''}"""

        # Build multimodal contents with chronological context
        contents = []

        # Add audio start timestamp context
        audio_start_ts = talk.get('audio_start_timestamp')
        if audio_start_ts:
            formatted_start = format_datetime_for_prompt(audio_start_ts)
            if formatted_start:
                contents.append(f"[Audio recording started at {formatted_start}]")

        contents.append(gemini_audio)

        # Add slides with photo metadata for chronological context
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

        gemini_model = get_gemini_model_for_multimodal(model)
        response = ai.models.generate_content(model=gemini_model, contents=contents)
        result = response.text

        save_ai_content(talk_id, "actions", result, gemini_model)
        return result

    except Exception as e:
        if on_fallback_warning:
            on_fallback_warning(f"Multimodal actions failed ({str(e)}), using text-only.")
        return extract_action_items(talk_id, model, custom_instructions)
