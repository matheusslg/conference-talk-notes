# Database layer - Supabase operations

from src.database.client import supabase

from src.database.talks import (
    create_talk,
    get_all_talks,
    get_talk_by_id,
    update_talk,
    delete_talk,
    update_talk_audio_url,
    update_talk_slide_urls,
)

from src.database.chunks import (
    get_talk_chunks,
    get_aligned_segments,
    get_uploaded_files,
    insert_chunk,
    delete_chunks_by_talk,
)

from src.database.ai_content import (
    save_ai_content,
    get_all_ai_content,
    delete_ai_content,
    get_ai_content,
    get_chat_history,
    save_chat_history,
)

from src.database.storage import (
    upload_audio,
    upload_slide,
)

__all__ = [
    # Client
    "supabase",
    # Talks
    "create_talk",
    "get_all_talks",
    "get_talk_by_id",
    "update_talk",
    "delete_talk",
    "update_talk_audio_url",
    "update_talk_slide_urls",
    # Chunks
    "get_talk_chunks",
    "get_aligned_segments",
    "get_uploaded_files",
    "insert_chunk",
    "delete_chunks_by_talk",
    # AI Content
    "save_ai_content",
    "get_all_ai_content",
    "delete_ai_content",
    "get_ai_content",
    "get_chat_history",
    "save_chat_history",
    # Storage
    "upload_audio",
    "upload_slide",
]
