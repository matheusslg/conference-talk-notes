# Insights layer - summary, quotes, actions, chat

from src.insights.summary import (
    fetch_from_storage,
    upload_audio_to_gemini_for_summary,
    generate_talk_summary,
    generate_talk_summary_multimodal,
    export_to_markdown,
)

from src.insights.quotes import (
    extract_key_quotes,
    extract_key_quotes_multimodal,
)

from src.insights.actions import (
    extract_action_items,
    extract_action_items_multimodal,
)

from src.insights.chat import (
    chat_with_talk,
)

__all__ = [
    # Summary
    "fetch_from_storage",
    "upload_audio_to_gemini_for_summary",
    "generate_talk_summary",
    "generate_talk_summary_multimodal",
    "export_to_markdown",
    # Quotes
    "extract_key_quotes",
    "extract_key_quotes_multimodal",
    # Actions
    "extract_action_items",
    "extract_action_items_multimodal",
    # Chat
    "chat_with_talk",
]
