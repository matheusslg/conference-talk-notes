# LLM layer - AI client initialization and text generation

from src.llm.clients import (
    gemini_ai,
    openai_ai,
    anthropic_ai,
    ai,
)

from src.llm.embeddings import generate_embedding

from src.llm.generation import (
    generate_with_llm,
    get_available_models,
)

__all__ = [
    # Clients
    "gemini_ai",
    "openai_ai",
    "anthropic_ai",
    "ai",
    # Embeddings
    "generate_embedding",
    # Generation
    "generate_with_llm",
    "get_available_models",
]
