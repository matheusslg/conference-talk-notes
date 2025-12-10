"""AI-generated content persistence (summaries, quotes, actions, chat)."""

import json
from src.database.client import supabase


def save_ai_content(talk_id: str, content_type: str, content: str, model: str):
    """Save AI-generated content (always creates new entry for history)."""
    supabase.from_("talk_ai_content").insert({
        "talk_id": talk_id,
        "content_type": content_type,
        "content": content,
        "model_used": model
    }).execute()


def get_all_ai_content(talk_id: str, content_type: str) -> list:
    """Get all stored AI content for a content type, ordered by newest first."""
    result = supabase.from_("talk_ai_content").select(
        "id, content, model_used, created_at"
    ).eq("talk_id", talk_id).eq("content_type", content_type).order("created_at", desc=True).execute()
    return result.data or []


def delete_ai_content(content_id: str):
    """Delete a specific AI content entry."""
    supabase.from_("talk_ai_content").delete().eq("id", content_id).execute()


def get_ai_content(talk_id: str, content_type: str) -> dict:
    """Get most recent AI content. Returns dict with 'content' and 'model_used' or None."""
    result = supabase.from_("talk_ai_content").select(
        "content, model_used, created_at"
    ).eq("talk_id", talk_id).eq("content_type", content_type).order("created_at", desc=True).limit(1).execute()
    return result.data[0] if result.data else None


def get_chat_history(talk_id: str) -> list:
    """Get stored chat history for a talk."""
    result = get_ai_content(talk_id, "chat")
    if result and result.get("content"):
        try:
            return json.loads(result["content"])
        except json.JSONDecodeError:
            return []
    return []


def save_chat_history(talk_id: str, history: list, model: str):
    """Save chat history as JSON."""
    save_ai_content(talk_id, "chat", json.dumps(history), model)
