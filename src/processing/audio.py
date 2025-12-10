"""Audio processing - transcription and conversion."""

import os
import json
import time
import tempfile
import subprocess
from datetime import datetime

from src.config import AUDIO_MIME_TYPES, AUDIO_FORMATS_NEED_CONVERSION
from src.llm import ai


def convert_audio_to_mp3(input_path: str) -> str:
    """Convert audio file to MP3 using ffmpeg.

    Returns path to converted MP3 file (caller must clean up).
    Raises exception if conversion fails.
    """
    output_path = tempfile.mktemp(suffix=".mp3")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-y", output_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr}")
        return output_path
    except FileNotFoundError:
        raise Exception("ffmpeg not found. Please install ffmpeg to convert audio files.")


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
                    value = value.replace('Z', '+00:00')
                    if '.' in value:
                        base = value.split('.')[0]
                        tz_part = value[value.rfind('+'):] if '+' in value else (value[value.rfind('-'):] if value.count('-') > 2 else '')
                        value = base + tz_part if tz_part else base
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is not None:
                        dt = dt.astimezone().replace(tzinfo=None)
                    return dt
                # Handle simple date format: "2024-12-04"
                if len(value) == 10 and value.count('-') == 2:
                    return datetime.strptime(value, "%Y-%m-%d")

        return None
    except Exception:
        return None


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe.

    Returns None if duration cannot be determined.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return None

        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        return None
    except Exception:
        return None


def get_audio_metadata(file_path: str, filename: str) -> dict:
    """Extract creation time and duration from audio file.

    Returns:
        {
            "filename": str,
            "creation_time": datetime | None,
            "duration_seconds": float | None
        }
    """
    return {
        "filename": filename,
        "creation_time": get_audio_creation_time(file_path),
        "duration_seconds": get_audio_duration(file_path)
    }


def sort_audio_fragments(audio_files: list[dict]) -> list[dict]:
    """Sort audio fragments by creation time and calculate offsets.

    Args:
        audio_files: List of {"bytes": bytes, "name": str}

    Returns:
        Sorted list with metadata:
        [
            {
                "bytes": bytes,
                "name": str,
                "creation_time": datetime | None,
                "duration_seconds": float | None,
                "offset_seconds": float,  # Cumulative offset from first file
                "gap_seconds": float | None  # Gap before this fragment (None for first)
            }
        ]
    """
    # Extract metadata for all files
    files_with_metadata = []
    for audio_file in audio_files:
        # Write to temp file to extract metadata
        ext = os.path.splitext(audio_file["name"])[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_file["bytes"])
            tmp_path = tmp.name

        try:
            metadata = get_audio_metadata(tmp_path, audio_file["name"])
            files_with_metadata.append({
                "bytes": audio_file["bytes"],
                "name": audio_file["name"],
                "creation_time": metadata["creation_time"],
                "duration_seconds": metadata["duration_seconds"],
            })
        finally:
            os.unlink(tmp_path)

    # Sort by creation time (files without timestamps go last, sorted by name)
    def sort_key(f):
        if f["creation_time"] is not None:
            return (0, f["creation_time"], f["name"])
        return (1, datetime.min, f["name"])

    sorted_files = sorted(files_with_metadata, key=sort_key)

    # Calculate cumulative offsets and gaps
    cumulative_offset = 0.0
    first_creation_time = None

    for i, f in enumerate(sorted_files):
        if i == 0:
            f["offset_seconds"] = 0.0
            f["gap_seconds"] = None
            first_creation_time = f["creation_time"]
        else:
            # Calculate gap from previous file
            prev = sorted_files[i - 1]
            if prev["creation_time"] and f["creation_time"] and prev["duration_seconds"]:
                # Gap = this file's start time - (previous file's start + duration)
                expected_end = prev["creation_time"].timestamp() + prev["duration_seconds"]
                actual_start = f["creation_time"].timestamp()
                gap = actual_start - expected_end
                f["gap_seconds"] = max(0.0, gap)  # Ignore negative gaps (overlap)
            else:
                f["gap_seconds"] = None

            # Offset = previous offset + previous duration + gap
            prev_duration = prev["duration_seconds"] or 0.0
            prev_gap = f["gap_seconds"] or 0.0
            cumulative_offset = prev["offset_seconds"] + prev_duration + prev_gap
            f["offset_seconds"] = cumulative_offset

    return sorted_files


def _parse_timestamp_as_hms(ts: str) -> int:
    """Parse timestamp as HH:MM:SS to seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _parse_timestamp_as_msf(ts: str) -> int:
    """Parse timestamp as MM:SS:FF (minutes:seconds:frames) to seconds."""
    parts = ts.split(":")
    # Ignore frames (last part), just use minutes and seconds
    return int(parts[0]) * 60 + int(parts[1])


def _format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    seconds = max(0, seconds)  # Ensure non-negative
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _detect_timestamp_format(segments: list[dict]) -> str:
    """Detect if timestamps are HH:MM:SS or MM:SS:FF format.

    Gemini inconsistently returns timestamps in different formats:
    - HH:MM:SS (hours:minutes:seconds) - standard format
    - MM:SS:FF (minutes:seconds:frames) - sometimes used for shorter audio

    Detection logic:
    - If the last segment's first component exceeds 59, it's likely minutes (MM:SS:FF)
    - If timestamps are clearly in hour range (>= 1 hour) for short audio, it's MM:SS:FF
    """
    if not segments:
        return "hms"

    # Check the last segment's start time
    last_start = segments[-1]["start"]
    parts = last_start.split(":")
    first_component = int(parts[0])

    # If first component > 59, it can't be hours in HH:MM:SS (max 23 for time of day)
    # But it also can't be minutes > 59 in standard format
    # So if > 59, it's likely MM:SS:FF where MM can go higher
    if first_component > 59:
        return "msf"

    # Check first segment - if it starts at 00:00:00, could be either format
    first_start = segments[0]["start"]
    first_parts = first_start.split(":")

    # If first segment starts with hour > 0 (like 16:30:00 or 22:00:00), it's wall-clock HH:MM:SS
    if int(first_parts[0]) > 0:
        return "hms"

    # For ambiguous cases (starts at 00:XX:XX), check if last timestamp
    # would make sense as minutes
    # e.g., "25:34:26" - 25 minutes makes sense, 25 hours doesn't for typical audio
    if first_component > 1 and first_component < 60:
        # Could be either 25 hours or 25 minutes
        # Assume MM:SS:FF if the value is reasonable for minutes (< 60)
        # and would be unreasonably long for hours
        return "msf"

    return "hms"


def normalize_transcript_timestamps(segments: list[dict]) -> list[dict]:
    """Normalize transcript timestamps to start from 00:00:00 in HH:MM:SS format.

    Handles two issues with Gemini's timestamp output:
    1. Wall-clock time (e.g., 22:15:30) instead of relative timestamps
    2. MM:SS:FF format (minutes:seconds:frames) instead of HH:MM:SS

    Args:
        segments: List of {"start": "XX:XX:XX", "end": "XX:XX:XX", "text": str}

    Returns:
        Segments with timestamps normalized to HH:MM:SS starting from 00:00:00
    """
    if not segments:
        return segments

    # Detect timestamp format
    ts_format = _detect_timestamp_format(segments)

    if ts_format == "msf":
        # Convert MM:SS:FF to HH:MM:SS
        normalized = []
        for seg in segments:
            start_sec = _parse_timestamp_as_msf(seg["start"])
            end_sec = _parse_timestamp_as_msf(seg["end"])
            normalized.append({
                "start": _format_timestamp(start_sec),
                "end": _format_timestamp(end_sec),
                "text": seg["text"]
            })
        return normalized

    # HH:MM:SS format - check if wall-clock time needs normalization
    first_start = _parse_timestamp_as_hms(segments[0]["start"])

    # If first segment starts at more than 1 hour, likely wall-clock time
    if first_start > 3600:
        normalized = []
        for seg in segments:
            start_sec = _parse_timestamp_as_hms(seg["start"]) - first_start
            end_sec = _parse_timestamp_as_hms(seg["end"]) - first_start
            normalized.append({
                "start": _format_timestamp(start_sec),
                "end": _format_timestamp(end_sec),
                "text": seg["text"]
            })
        return normalized

    return segments


def _parse_timestamp(ts: str) -> int:
    """Parse HH:MM:SS timestamp to seconds. Use after normalization."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def merge_transcripts(transcripts_with_offsets: list[dict]) -> list[dict]:
    """Merge transcripts from multiple audio fragments with timestamp adjustments.

    Args:
        transcripts_with_offsets: List of {
            "segments": [{"start": "HH:MM:SS", "end": "HH:MM:SS", "text": str}],
            "offset_seconds": float,
            "filename": str
        }

    Returns:
        Combined segments with adjusted timestamps in HH:MM:SS format
    """
    merged = []

    for transcript in transcripts_with_offsets:
        offset = transcript["offset_seconds"]
        filename = transcript.get("filename", "unknown")

        # Normalize timestamps first (handles Gemini returning wall-clock time)
        normalized_segments = normalize_transcript_timestamps(transcript["segments"])

        for segment in normalized_segments:
            start_seconds = _parse_timestamp(segment["start"])
            end_seconds = _parse_timestamp(segment["end"])

            # Apply offset
            new_start = start_seconds + offset
            new_end = end_seconds + offset

            merged.append({
                "start": _format_timestamp(new_start),
                "end": _format_timestamp(new_end),
                "text": segment["text"],
                "source_file": filename
            })

    return merged


def get_gemini_model_for_multimodal(model: str) -> str:
    """Map model preference to Gemini model for multimodal requests.

    For multimodal (audio/image), we must use Gemini.
    This maps the user's preference appropriately.
    """
    if model.startswith("gemini"):
        return model
    # Default to flash for non-Gemini preferences
    return "gemini-2.5-flash"


def transcribe_audio_with_timestamps(file_path: str, model: str = "gemini-2.5-pro") -> list[dict]:
    """Transcribe audio and return segments with timestamps.

    Returns: [{"start": "00:01:30", "end": "00:02:45", "text": "..."}, ...]
    """
    ext = os.path.splitext(file_path)[1].lower()
    converted_path = None

    # Convert unsupported formats to MP3
    if ext in AUDIO_FORMATS_NEED_CONVERSION:
        converted_path = convert_audio_to_mp3(file_path)
        file_path = converted_path
        ext = ".mp3"

    try:
        # Get MIME type for the audio file
        mime_type = AUDIO_MIME_TYPES.get(ext, "audio/mpeg")
        uploaded_file = ai.files.upload(file=file_path, config={"mime_type": mime_type})

        # Wait for file to be processed
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

        gemini_model = get_gemini_model_for_multimodal(model)
        response = ai.models.generate_content(
            model=gemini_model,
            contents=[uploaded_file, prompt]
        )

        text = response.text.strip()
        # Clean up markdown if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(text)
    finally:
        # Clean up converted file if we created one
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)
