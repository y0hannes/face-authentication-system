import streamlit as st
from app.utils import init_session_state, list_registered_users, user_image_count


def show():
    init_session_state()

    st.title("Register New User")
    st.markdown("Capture your face images and train the model.")
    st.divider()

    # ── Username input ──────────────────────────────────────────────────────
    username = st.text_input("Username", placeholder="e.g. alice")

    if username:
        username = username.strip()
        if len(username) < 2:
            st.error("Username must be at least 2 characters.")
            return
        if " " in username:
            st.info("Spaces will be replaced with underscores.")

        existing_count = user_image_count(username)
        if existing_count > 0:
            st.warning(
                f"'{username}' already has {existing_count} image(s). "
                "Capturing again will add more images. Retrain after capturing."
            )

    st.divider()
    st.info("📸 Capture and 🧠 Train sections coming next.")

    # ── Sidebar: registered users ───────────────────────────────────────────
    st.sidebar.markdown("### Registered Users")
    users = list_registered_users()
    if users:
        for u in users:
            count = user_image_count(u)
            st.sidebar.write(f"👤 **{u}** — {count} image(s)")
    else:
        st.sidebar.caption("No users registered yet.")
