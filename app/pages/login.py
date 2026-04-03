import streamlit as st
from app.utils import init_session_state


def show():
    init_session_state()

    st.title("Login — Face Authentication")
    st.info("🔒 Login functionality coming soon. Please register users and train the model first.")
