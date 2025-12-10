"""Shared utility functions for text processing and formatting."""

from datetime import datetime, timedelta


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start + overlap >= len(text):
            break
    return chunks


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


def format_datetime_for_prompt(dt_input) -> str:
    """Format datetime to user-friendly string for multimodal prompts.

    Accepts ISO string or datetime object.
    Returns: 'YYYY-MM-DD HH:MM:SS' format.
    """
    if dt_input is None:
        return None
    try:
        if isinstance(dt_input, str):
            dt = datetime.fromisoformat(dt_input.replace('Z', '+00:00'))
        else:
            dt = dt_input
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_input) if dt_input else None


def format_timestamped_segments(segments: list) -> str:
    """Format audio segments with timestamps for storage and display.

    Args:
        segments: List of {"start": "00:00:00", "end": "00:01:23", "text": "..."} dicts

    Returns:
        Formatted string with [MM:SS] prefixes
    """
    lines = []
    for seg in segments:
        start_seconds = parse_timestamp_to_seconds(seg["start"])
        timestamp = format_seconds_to_timestamp(start_seconds)
        lines.append(f"[{timestamp}] {seg['text']}")
    return "\n".join(lines)
