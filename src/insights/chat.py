"""RAG-based chat with talk content."""

from src.llm import generate_with_llm
from src.search import search_talk_content


def chat_with_talk(talk_id: str, question: str, chat_history: list, model: str = "gemini-2.5-pro") -> str:
    """Answer questions about this specific talk using RAG.

    Args:
        talk_id: Talk identifier
        question: User's question
        chat_history: List of previous exchanges [{"user": "...", "assistant": "..."}]
        model: Model to use for generation

    Returns:
        Generated answer based on talk content
    """
    # Search for relevant context
    results = search_talk_content(question, talk_id=talk_id, match_count=5)

    if not results:
        return "I couldn't find relevant information in this talk to answer your question. Try rephrasing or ask something else about the talk content."

    context = "\n\n".join([
        f"[{r['content_type']}] {r['content']}"
        for r in results
    ])

    # Build chat history context
    history_text = ""
    if chat_history:
        history_text = "\n\n**Previous conversation:**\n"
        for msg in chat_history[-6:]:  # Last 3 exchanges
            history_text += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"

    prompt = f"""You are a helpful assistant answering questions about a specific AWS re:Invent talk.

**Relevant content from the talk:**
{context}
{history_text}
**Current question:** {question}

**Instructions:**
- Answer based ONLY on the provided talk content
- Be specific and reference the talk material
- If the answer isn't in the content, say so
- Keep responses concise but helpful
- Use markdown formatting"""

    return generate_with_llm(prompt, model)
