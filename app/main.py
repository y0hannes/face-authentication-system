import sys
import os

# Ensure project root is on sys.path regardless of where Streamlit is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from app.pages import register, login
from src.predict import is_model_ready

#  Page config
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="expanded",
)

#  Sidebar navigation 
st.sidebar.title("🔐 Face Auth")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Register", "Login"],
    help="Register to capture your face data and train the model. "
         "Login to authenticate using your face.",
)

# Model readiness badge in sidebar
if is_model_ready():
    st.sidebar.success("✅ Model is ready")
else:
    st.sidebar.warning("⚠️ Model not trained yet")

st.sidebar.markdown("---")
st.sidebar.caption("Face Authentication System — v1.0")

#  Page routing
if page == "Register":
    register.show()
elif page == "Login":
    login.show()