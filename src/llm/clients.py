"""AI client initialization for Gemini, OpenAI, and Anthropic."""

import os
import streamlit as st
from google import genai
from openai import OpenAI
import anthropic


@st.cache_resource
def init_ai_clients():
    """Initialize AI clients with API keys from env or secrets."""
    gemini_client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    )

    # OpenAI client (optional)
    openai_key = os.environ.get("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    openai_client = OpenAI(api_key=openai_key) if openai_key else None

    # Anthropic client (optional)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", st.secrets.get("ANTHROPIC_API_KEY", ""))
    anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None

    return gemini_client, openai_client, anthropic_client


# Initialize clients
gemini_ai, openai_ai, anthropic_ai = init_ai_clients()

# Alias for backward compatibility
ai = gemini_ai
