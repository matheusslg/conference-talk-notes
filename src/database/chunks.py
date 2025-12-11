"""Talk chunks operations - segments, transcripts, slides."""

from src.database.client import supabase, with_retry


@with_retry()
def get_talk_chunks(talk_id: str) -> list:
    """Get all chunks for a talk, ordered by creation time."""
    result = supabase.from_("talk_chunks").select("*").eq("talk_id", talk_id).order("created_at").execute()
    return result.data or []


@with_retry()
def get_aligned_segments(talk_id: str) -> list:
    """Get aligned segments with slide numbers and timestamps, ordered by slide number."""
    result = supabase.from_("talk_chunks").select(
        "slide_number, start_time_seconds, end_time_seconds, content"
    ).eq("talk_id", talk_id).eq("content_type", "aligned_segment").order("slide_number").execute()
    return result.data or []


@with_retry()
def get_uploaded_files(talk_id: str) -> dict:
    """Get list of uploaded files grouped by type."""
    chunks = supabase.from_("talk_chunks").select(
        "source_file, content_type, created_at"
    ).eq("talk_id", talk_id).order("created_at").execute()

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


@with_retry()
def insert_chunk(
    talk_id: str,
    content_type: str,
    content: str,
    source_file: str = None,
    embedding: list = None,
    slide_number: int = None,
    slide_url: str = None,
    slide_thumbnail: str = None,
    start_time_seconds: float = None,
    end_time_seconds: float = None
):
    """Insert a new chunk into talk_chunks table."""
    data = {
        "talk_id": talk_id,
        "content_type": content_type,
        "content": content,
    }
    if source_file:
        data["source_file"] = source_file
    if embedding:
        data["embedding"] = embedding
    if slide_number is not None:
        data["slide_number"] = slide_number
    if slide_url:
        data["slide_url"] = slide_url
    if slide_thumbnail:
        data["slide_thumbnail"] = slide_thumbnail
    if start_time_seconds is not None:
        data["start_time_seconds"] = start_time_seconds
    if end_time_seconds is not None:
        data["end_time_seconds"] = end_time_seconds

    supabase.from_("talk_chunks").insert(data).execute()


@with_retry()
def delete_chunks_by_talk(talk_id: str):
    """Delete all chunks for a talk."""
    supabase.from_("talk_chunks").delete().eq("talk_id", talk_id).execute()
