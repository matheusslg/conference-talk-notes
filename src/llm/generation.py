"""LLM text generation abstraction across providers."""

from src.config import LLM_MODELS
from src.llm.clients import gemini_ai, openai_ai, anthropic_ai


def generate_with_llm(prompt: str, model: str) -> str:
    """Generate text using the specified LLM model."""
    if model.startswith("gemini"):
        response = gemini_ai.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    elif model.startswith("gpt"):
        if not openai_ai:
            raise ValueError("OpenAI API key not configured")
        response = openai_ai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    elif model.startswith("claude"):
        if not anthropic_ai:
            raise ValueError("Anthropic API key not configured")
        response = anthropic_ai.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unknown model: {model}")


def get_available_models() -> list:
    """Return list of models that have API keys configured."""
    available = LLM_MODELS["Gemini"].copy()  # Gemini always available
    if openai_ai:
        available.extend(LLM_MODELS["OpenAI"])
    if anthropic_ai:
        available.extend(LLM_MODELS["Anthropic"])
    return available
