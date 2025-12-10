"""Audio-slide alignment functions."""

import json
from datetime import datetime, timedelta

from src.config import AI_ALIGNMENT_PROMPT, PHOTO_CAPTURE_DELAY_SECONDS
from src.utils import (
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
    format_timestamped_segments,
)
from src.database import get_aligned_segments
from src.llm import ai
from src.processing.audio import get_gemini_model_for_multimodal


def build_slide_timing_context(talk_id: str, audio_start_timestamp: str = None) -> str:
    """Build a timing context string for multimodal prompts.

    Returns a formatted string showing when each slide appears in the recording,
    with both relative timestamps (MM:SS) and real-world times if available.
    """
    segments = get_aligned_segments(talk_id)
    if not segments:
        return ""

    # Parse audio start time if available
    audio_start = None
    if audio_start_timestamp:
        try:
            audio_start = datetime.fromisoformat(audio_start_timestamp.replace('Z', '+00:00'))
        except:
            pass

    lines = ["**Slide Timeline:**"]
    for seg in segments:
        slide_num = seg.get('slide_number', 0)
        start_sec = seg.get('start_time_seconds')
        end_sec = seg.get('end_time_seconds')

        if start_sec is None:
            continue

        # Format relative timestamp
        rel_start = format_seconds_to_timestamp(start_sec)
        rel_end = format_seconds_to_timestamp(end_sec) if end_sec else "end"

        # Calculate real-world time if audio_start is available
        if audio_start:
            real_start = audio_start + timedelta(seconds=start_sec)
            real_time_str = real_start.strftime("%H:%M:%S")
            lines.append(f"- Slide {slide_num}: {rel_start} - {rel_end} (real time: {real_time_str})")
        else:
            lines.append(f"- Slide {slide_num}: {rel_start} - {rel_end}")

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


def align_slides_with_ai(audio_segments: list, slides_data: list, model: str = "gemini-2.5-pro") -> list:
    """Use Gemini to semantically align slides with transcript segments.

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

    gemini_model = get_gemini_model_for_multimodal(model)
    response = ai.models.generate_content(
        model=gemini_model,
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
    """Distribute audio segments evenly across slides as a fallback.

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
