"""Embedding generation using Gemini."""

from src.llm.clients import ai


def generate_embedding(text: str) -> list:
    """Generate text embedding using Gemini's embedding model."""
    response = ai.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embeddings[0].values
