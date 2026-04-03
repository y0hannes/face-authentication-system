import os
import sys

# sys.path must be fixed BEFORE any local-package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402
from app.pages import register, login  # noqa: E402
from src.predict import is_model_ready  # noqa: E402
from app.utils import list_registered_users  # noqa: E402

# Page config
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("🔐 Face Auth")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Register", "Login"],
    help="Register to capture your face data and train the model. "
         "Login to authenticate using your face.",
)

# Model readiness badge
if is_model_ready():
    st.sidebar.success("✅ Model is ready")
else:
    st.sidebar.warning("⚠️ Model not trained yet")

# Registered user count
users = list_registered_users()
st.sidebar.metric("Registered Users", len(users))

st.sidebar.markdown("### User Details")
if users:
    from app.utils import user_image_count
    for u in users:
        count = user_image_count(u)
        st.sidebar.write(f"👤 **{u}** — {count} image(s)")
else:
    st.sidebar.caption("No users registered yet.")

st.sidebar.markdown("---")
st.sidebar.caption("Face Authentication System — v1.0")

# Page routing
try:
    if page == "Register":
        register.show()
    elif page == "Login":
        login.show()
except Exception as e:
    st.error(f"Page error: {e}")
