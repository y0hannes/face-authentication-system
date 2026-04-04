import os
import sys

# sys.path must be fixed BEFORE any local-package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402
from app.pages import register, login, insights  # noqa: E402
from src.predict import is_model_ready  # noqa: E402
from app.utils import (  # noqa: E402
    list_registered_users,
    user_image_count,
    delete_user,
    get_login_attempts,
    get_failed_attempts,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
        }

        /* Hide Streamlit's default page nav */
        [data-testid="stSidebarNav"] { display: none; }

        /* Sidebar chrome */
        [data-testid="stSidebar"] {
            background-color: #f2f4f8 !important;
            border-right: 1px solid rgba(228, 189, 186, 0.15);
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

        /* ── Radio nav: hide the native widget label & dot, style each option ── */
        [data-testid="stSidebar"] .stRadio > label { display: none; }
        [data-testid="stSidebar"] .stRadio > div {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        [data-testid="stSidebar"] .stRadio > div > label {
            display: flex !important;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            color: #64748b;
            transition: background 0.15s, transform 0.15s, color 0.15s;
            background: transparent;
        }
        [data-testid="stSidebar"] .stRadio > div > label:hover {
            background: rgba(203, 213, 225, 0.4);
            transform: translateX(4px);
            color: #475569;
        }
        [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
            background: #ffffff;
            color: #b81120;
            font-weight: 700;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            transform: translateX(0);
        }
        [data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
        [data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none; }

        /* ── Delete buttons inside user list ── */
        [data-testid="stSidebar"] div.stButton > button {
            background: transparent;
            border: none;
            color: #b81120;
            padding: 2px 6px;
            font-size: 0.75rem;
            border-radius: 4px;
            transition: background 0.15s;
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(184,17,32,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# ── Load live data ────────────────────────────────────────────────────────────
users        = list_registered_users()
total_users  = len(users)
login_count  = get_login_attempts()
failed_count = get_failed_attempts()
model_ok     = is_model_ready()

# ── Sidebar: Brand header ─────────────────────────────────────────────────────
st.sidebar.markdown("""
    <div style="padding: 20px 4px 16px; border-bottom: 1px solid rgba(228,189,186,0.2); margin-bottom: 8px;">
        <h2 style="font-size: 1.125rem; font-weight: 900; color: #1e293b;
                   margin: 0; font-family: Inter, sans-serif;">Auth Hub</h2>
        <p style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.2em;
                  color: #b81120; font-weight: 700; margin: 3px 0 0;">Precision Security</p>
    </div>
""", unsafe_allow_html=True)

# ── Sidebar: Navigation ───────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Register"

_pages = ["Register", "Login", "Insights"]
_icons = {"Register": "person_add", "Login": "fingerprint", "Insights": "insights"}

selected = st.sidebar.radio(
    "nav",
    options=_pages,
    index=_pages.index(st.session_state.current_page),
    label_visibility="collapsed",
    format_func=lambda p: f"{p}",   # plain text; icon injected via CSS ::before workaround below
)

if selected != st.session_state.current_page:
    st.session_state.current_page = selected
    st.rerun()

page = st.session_state.current_page

# ── Sidebar: Live stats strip ─────────────────────────────────────────────────
model_dot  = "#008472" if model_ok  else "#ba1a1a"
model_txt  = "Model ready" if model_ok else "Not trained"
failed_color = "#ba1a1a" if failed_count > 0 else "#008472"

st.sidebar.markdown(f"""
    <div style="margin: 16px 0; border-radius: 10px; background: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06); overflow: hidden;">

      <!-- Enrolled -->
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding: 10px 14px; border-bottom: 1px solid #eceef2;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="material-symbols-outlined" style="font-size:16px; color:#64748b;">group</span>
          <span style="font-size:12px; color:#64748b; font-weight:500; font-family:Inter,sans-serif;">Enrolled</span>
        </div>
        <span style="font-size:13px; font-weight:800; color:#191c1f;
                     font-family:Inter,sans-serif;">{total_users}</span>
      </div>

      <!-- Login attempts -->
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding: 10px 14px; border-bottom: 1px solid #eceef2;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="material-symbols-outlined" style="font-size:16px; color:#64748b;">login</span>
          <span style="font-size:12px; color:#64748b; font-weight:500; font-family:Inter,sans-serif;">Logins</span>
        </div>
        <span style="font-size:13px; font-weight:800; color:#191c1f;
                     font-family:Inter,sans-serif;">{login_count:,}</span>
      </div>

      <!-- Failed attempts -->
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding: 10px 14px; border-bottom: 1px solid #eceef2;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="material-symbols-outlined" style="font-size:16px; color:{failed_color};">warning</span>
          <span style="font-size:12px; color:#64748b; font-weight:500; font-family:Inter,sans-serif;">Failed</span>
        </div>
        <span style="font-size:13px; font-weight:800; color:{failed_color};
                     font-family:Inter,sans-serif;">{failed_count:,}</span>
      </div>

      <!-- Model status -->
      <div style="display:flex; align-items:center; justify-content:space-between;
                  padding: 10px 14px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="material-symbols-outlined" style="font-size:16px; color:{model_dot};">smart_toy</span>
          <span style="font-size:12px; color:#64748b; font-weight:500; font-family:Inter,sans-serif;">Model</span>
        </div>
        <span style="display:inline-flex; align-items:center; gap:4px; font-size:11px;
                     font-weight:700; color:{model_dot}; font-family:Inter,sans-serif;">
          <span style="width:6px; height:6px; border-radius:50%;
                       background:{model_dot}; display:inline-block;"></span>
          {model_txt}
        </span>
      </div>

    </div>
""", unsafe_allow_html=True)

# ── Sidebar: User details (keep functional — with delete) ─────────────────────
st.sidebar.markdown("""
    <p style="font-size: 10px; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.15em; color: #94a3b8; margin: 4px 0 8px;
              font-family: Inter, sans-serif; padding: 0 2px;">
      Registered Users
    </p>
""", unsafe_allow_html=True)

if users:
    for u in users:
        count = user_image_count(u)
        col1, col2 = st.sidebar.columns([5, 1])
        col1.markdown(
            f'<div style="font-size:0.8rem; padding: 5px 2px; font-family:Inter,sans-serif;">'
            f'<span style="font-weight:600; color:#1e293b;">{u}</span>'
            f'<span style="color:#94a3b8; font-size:11px;"> · {count} img</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if col2.button("🗑️", key=f"del_{u}", help=f"Delete {u}"):
            delete_user(u)
            st.rerun()
else:
    st.sidebar.markdown(
        '<p style="font-size:12px; color:#94a3b8; font-style:italic; padding: 4px 2px; '
        'font-family:Inter,sans-serif;">No users registered yet.</p>',
        unsafe_allow_html=True,
    )

# ── Sidebar: Footer ───────────────────────────────────────────────────────────
st.sidebar.markdown("""
    <div style="margin-top: 20px; padding-top: 14px; border-top: 1px solid rgba(228,189,186,0.2);">
      <p style="font-size: 10px; color: #cbd5e1; font-family: Inter, sans-serif;
                text-align: center; margin: 0;">Face Auth System · v1.0</p>
    </div>
""", unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
try:
    if page == "Register":
        register.show()
    elif page == "Login":
        login.show()
    elif page == "Insights":
        insights.show()
except Exception as e:
    st.error(f"Page error: {e}")
