"""Vector search functionality for talk content."""

from src.database import supabase
from src.llm import generate_embedding, ai
from src.processing import get_gemini_model_for_multimodal


def search_talk_content(
    query: str,
    talk_id: str = None,
    content_types: list = None,
    match_count: int = 10
) -> list:
    """Search talk content using vector similarity.

    Args:
        query: Search query text
        talk_id: Optional filter by talk ID
        content_types: Optional filter by content types
        match_count: Max results to return

    Returns:
        List of matching chunks with similarity scores
    """
    query_vector = generate_embedding(query)

    params = {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": match_count,
    }

    if talk_id:
        params["filter_talk_id"] = talk_id
    if content_types:
        params["filter_content_types"] = content_types

    result = supabase.rpc("search_talk_chunks", params).execute()
    return result.data or []


def generate_search_insights(query: str, results: list, model: str = "gemini-2.5-pro") -> str:
    """Generate AI insights from search results.

    Args:
        query: Original search query
        results: Search results from search_talk_content
        model: LLM model to use

    Returns:
        Markdown-formatted insights
    """
    if not results:
        return ""

    audio_context = [r for r in results if r['content_type'] == 'audio_transcript']
    ocr_context = [r for r in results if r['content_type'] == 'slide_ocr']
    vision_context = [r for r in results if r['content_type'] == 'slide_vision']

    context_parts = []
    if audio_context:
        context_parts.append("**From Audio Transcripts:**\n" + "\n\n".join([r['content'] for r in audio_context[:3]]))
    if ocr_context:
        context_parts.append("**From Slide Text:**\n" + "\n\n".join([r['content'] for r in ocr_context[:3]]))
    if vision_context:
        context_parts.append("**From Slide Visuals:**\n" + "\n\n".join([r['content'] for r in vision_context[:3]]))

    context = "\n\n---\n\n".join(context_parts)

    gemini_model = get_gemini_model_for_multimodal(model)
    response = ai.models.generate_content(
        model=gemini_model,
        contents=f"""Based on the user's question and the relevant content from conference talk materials below, provide a comprehensive answer.

**User's Question:** {query}

**Content from Talk Materials:**
{context}

**Instructions:**
- Synthesize information from audio transcripts AND slide content
- Reference specific slides when relevant (e.g., "As shown in slide 3...")
- Be technical and specific for AWS re:Invent content
- Use bullet points for clarity

Provide your response in markdown format."""
    )
    return response.text
