"""Password authentication for the application."""

import streamlit as st
from src.config import DEFAULT_EVENT


def check_password() -> bool:
    """Returns True if user entered correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Conference Talk Notes")
    st.caption(DEFAULT_EVENT)
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    # Autofocus password field
    st.markdown('''
    <script>
        // Autofocus password input
        setTimeout(() => {
            const pwInput = document.querySelector('input[type="password"]');
            if (pwInput) pwInput.focus();
        }, 100);
    </script>
    ''', unsafe_allow_html=True)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password")
    return False
