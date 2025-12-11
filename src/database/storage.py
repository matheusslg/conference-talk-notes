"""Supabase storage operations for audio and slide files."""

from src.database.client import supabase, with_retry


@with_retry()
def upload_audio(filename: str, file_bytes: bytes) -> str:
    """Upload audio file to storage and return public URL."""
    supabase.storage.from_("talk-audio").upload(
        filename, file_bytes, {"content-type": "audio/mpeg"}
    )
    return supabase.storage.from_("talk-audio").get_public_url(filename)


@with_retry()
def upload_slide(filename: str, file_bytes: bytes, content_type: str = "image/jpeg") -> str:
    """Upload slide image to storage and return public URL."""
    supabase.storage.from_("talk-slides").upload(
        filename, file_bytes, {"content-type": content_type}
    )
    return supabase.storage.from_("talk-slides").get_public_url(filename)
