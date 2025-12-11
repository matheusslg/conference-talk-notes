"""Supabase client initialization with retry logic."""

import os
import time
import functools
import streamlit as st
from supabase import create_client


@st.cache_resource
def get_supabase_client():
    """Initialize and return Supabase client."""
    return create_client(
        os.environ.get("SUPABASE_URL", st.secrets.get("SUPABASE_URL", "")),
        os.environ.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_SERVICE_KEY", ""))
    )


def with_retry(max_retries: int = 3, delay: float = 0.5):
    """Decorator to retry database operations on transient network errors.

    Catches httpx.ReadError and similar connection issues, retrying with
    exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_type = type(e).__name__
                    # Retry on network/connection errors
                    if "ReadError" in error_type or "ConnectError" in error_type or "TimeoutException" in error_type:
                        last_error = e
                        if attempt < max_retries - 1:
                            time.sleep(delay * (2 ** attempt))  # Exponential backoff
                            continue
                    raise
            raise last_error
        return wrapper
    return decorator


# Global client instance
supabase = get_supabase_client()
