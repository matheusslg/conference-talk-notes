"""Talk CRUD operations."""

from src.database.client import supabase


def create_talk(title: str, speaker: str = None, author_name: str = None) -> str:
    """Create a new talk and return its ID."""
    result = supabase.from_("talks").insert({
        "title": title,
        "speaker": speaker,
        "author_name": author_name,
    }).execute()
    return result.data[0]["id"] if result.data else None


def get_all_talks() -> list:
    """Get all talks with enriched metadata."""
    result = supabase.from_("talks").select("*").order("created_at", desc=True).execute()
    talks = result.data or []

    # Enrich with segment count and first thumbnail
    for talk in talks:
        chunks = supabase.from_("talk_chunks").select(
            "content_type, slide_thumbnail, slide_number"
        ).eq("talk_id", talk["id"]).order("slide_number").execute()
        chunk_data = chunks.data or []

        # Count aligned segments (new format) or fall back to legacy counts
        aligned = len([c for c in chunk_data if c["content_type"] == "aligned_segment"])
        legacy = len([c for c in chunk_data if c["content_type"] in ("audio_transcript", "slide_ocr", "slide_vision")])
        talk["segment_count"] = aligned or legacy

        # Get first thumbnail
        thumbnails = [c.get("slide_thumbnail") for c in chunk_data if c.get("slide_thumbnail")]
        talk["first_thumbnail"] = thumbnails[0] if thumbnails else None

        # Check if has summary (processed indicator)
        ai_content = supabase.from_("talk_ai_content").select("content_type").eq("talk_id", talk["id"]).execute()
        talk["has_summary"] = any(c["content_type"] == "summary" for c in (ai_content.data or []))

    return talks


def get_talk_by_id(talk_id: str) -> dict:
    """Get a single talk by ID."""
    result = supabase.from_("talks").select("*").eq("id", talk_id).single().execute()
    return result.data


def update_talk(talk_id: str, title: str, speaker: str = None, author_name: str = None) -> bool:
    """Update talk metadata."""
    supabase.from_("talks").update({
        "title": title,
        "speaker": speaker,
        "author_name": author_name,
        "updated_at": "now()"
    }).eq("id", talk_id).execute()
    return True


def delete_talk(talk_id: str) -> bool:
    """Delete a talk and all associated data (via cascade)."""
    supabase.from_("talks").delete().eq("id", talk_id).execute()
    return True


def update_talk_audio_url(talk_id: str, audio_url: str, transcript: str = None):
    """Update talk with audio URL and optionally transcript."""
    update_data = {"audio_url": audio_url}
    if transcript:
        update_data["transcript"] = transcript
    supabase.from_("talks").update(update_data).eq("id", talk_id).execute()


def update_talk_slide_urls(talk_id: str, slide_urls: list):
    """Update talk with slide URLs."""
    supabase.from_("talks").update({"slide_urls": slide_urls}).eq("id", talk_id).execute()
