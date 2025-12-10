"""Password authentication for the application with URL-based persistence."""

import hashlib
import streamlit as st
from src.config import DEFAULT_EVENT


def hash_token(password: str) -> str:
    """Create a short hash token for URL storage."""
    return hashlib.sha256(password.encode()).hexdigest()[:12]


def check_password() -> bool:
    """Returns True if user entered correct password.

    Uses URL query params to persist authentication across page refreshes.
    """
    app_password = st.secrets.get("APP_PASSWORD", "")
    expected_token = hash_token(app_password)

    # Check if already authenticated via session state
    if st.session_state.get("password_correct", False):
        return True

    # Check URL query param for stored auth token
    auth_token = st.query_params.get("auth")
    if auth_token == expected_token:
        st.session_state["password_correct"] = True
        return True

    def password_entered():
        entered = st.session_state.get("password", "")
        if entered == app_password:
            st.session_state["password_correct"] = True
            # Store auth token in URL query params
            st.query_params["auth"] = expected_token
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    st.title("Conference Talk Notes")
    st.caption(DEFAULT_EVENT)
    st.text_input("Password", type="password", on_change=password_entered, key="password")

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")
    return False


def get_user_name() -> str | None:
    """Get user name from URL query params or session state."""
    # Check session state first
    if st.session_state.get("current_user_name"):
        return st.session_state.current_user_name

    # Check URL query params
    stored_name = st.query_params.get("user")
    if stored_name:
        st.session_state.current_user_name = stored_name
        return stored_name

    return None


def set_user_name(name: str):
    """Set user name in session state and URL query params."""
    st.session_state.current_user_name = name
    # Store in URL query params
    st.query_params["user"] = name


def logout():
    """Clear authentication state."""
    st.session_state["password_correct"] = False
    st.session_state.current_user_name = None
    # Clear query params
    if "auth" in st.query_params:
        del st.query_params["auth"]
    if "user" in st.query_params:
        del st.query_params["user"]
