"""Supabase client initialization."""

import os
import streamlit as st
from supabase import create_client


@st.cache_resource
def get_supabase_client():
    """Initialize and return Supabase client."""
    return create_client(
        os.environ.get("SUPABASE_URL", st.secrets.get("SUPABASE_URL", "")),
        os.environ.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_SERVICE_KEY", ""))
    )


# Global client instance
supabase = get_supabase_client()
