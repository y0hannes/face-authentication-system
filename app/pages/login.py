import time
import datetime
import cv2
import streamlit as st

from src.predict import is_model_ready, predict_face
from app.utils import init_session_state, log_auth_event, get_auth_events
from config import THRESHOLD


def _file_to_cv2(uploaded_file):
    """Convert a Streamlit uploaded file (camera_input) to an OpenCV BGR image."""
    import numpy as np
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)


# ── Page CSS ──────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] { background: var(--bg-page); color: var(--txt-main); }
[data-testid="block-container"]    { padding-top: 1.5rem !important; }

/* Hero title */
.login-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--txt-main);
    margin: 0 0 12px;
    line-height: 1.1;
    font-family: Inter, sans-serif;
}
.login-title-bar {
    height: 4px;
    width: 80px;
    background: linear-gradient(to right, var(--brand), #dc3135);
    border-radius: 9999px;
    margin-bottom: 28px;
}

/* Stat micro-cards */
.stat-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
}
.stat-card {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
}
.stat-card-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
    display: block;
}
.stat-card-value {
    font-size: 1.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: var(--txt-main);
    font-family: Inter, sans-serif;
    line-height: 1;
}
.stat-card-unit {
    font-size: 11px;
    color: var(--txt-sub);
    font-weight: 500;
    margin-left: 2px;
}

/* Auth panel */
.auth-panel {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 28px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}
.auth-panel-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--txt-main);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    font-family: Inter, sans-serif;
}
.auth-panel-title-icon {
    color: var(--brand);
    font-size: 22px;
}
.auth-field-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--txt-sub);
    margin-bottom: 6px;
    display: block;
}
.auth-field-value {
    width: 100%;
    background: var(--bg-input);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: var(--txt-sub);
    font-family: monospace;
    border: 1px solid var(--border);
    margin-bottom: 18px;
}
.auth-hint {
    font-size: 13px;
    color: var(--txt-sub);
    line-height: 1.6;
    margin-bottom: 14px;
    font-family: Inter, sans-serif;
}
.auth-info-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: rgba(0,132,114,0.08);
    border-radius: 8px;
    margin-bottom: 20px;
}
.auth-info-chip-icon { font-size: 18px; color: #008472; }
.auth-info-chip-text { font-size: 12px; color: #008472; font-weight: 600; font-family: Inter, sans-serif; }

/* Recent sessions card */
.sessions-card {
    background: var(--bg-input);
    border-radius: 12px;
    padding: 20px;
    margin-top: 14px;
}
.sessions-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--txt-sub);
    margin-bottom: 12px;
    display: block;
    font-family: Inter, sans-serif;
}
.session-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    background: var(--bg-card);
    border-radius: 8px;
    margin-bottom: 8px;
}
.session-row:last-child { margin-bottom: 0; }
.session-left { display: flex; align-items: center; gap: 10px; }
.session-dot-green { width: 8px; height: 8px; border-radius: 50%; background: #008472; flex-shrink: 0; }
.session-dot-red   { width: 8px; height: 8px; border-radius: 50%; background: #ba1a1a; flex-shrink: 0; }
.session-name { font-size: 12px; font-weight: 600; color: var(--txt-main); font-family: Inter, sans-serif; }
.session-time { font-size: 10px; color: var(--txt-sub); font-family: Inter, sans-serif; }
.sessions-empty {
    font-size: 12px;
    color: var(--txt-sub);
    text-align: center;
    font-style: italic;
    padding: 8px 0;
    font-family: Inter, sans-serif;
}
</style>
"""


def _time_ago(ts_str: str) -> str:
    """Convert an ISO timestamp to a human-friendly 'X ago' string."""
    try:
        ts = datetime.datetime.fromisoformat(ts_str)
        delta = datetime.datetime.now() - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        elif secs < 3600:
            return f"{secs // 60}m ago"
        elif secs < 86400:
            return f"{secs // 3600}h ago"
        else:
            return f"{secs // 86400}d ago"
    except Exception:
        return ""


def _recent_sessions_html(events: list[dict]) -> str:
    if not events:
        return '<div class="sessions-empty">No recent sessions</div>'
    rows = ""
    for e in events:
        is_auth = e.get("status") == "authenticated"
        dot_cls = "session-dot-green" if is_auth else "session-dot-red"
        name = e.get("username", "Unknown")
        ago = _time_ago(e.get("ts", ""))
        rows += f"""
        <div class="session-row">
          <div class="session-left">
            <div class="{dot_cls}"></div>
            <span class="session-name">{name}</span>
          </div>
          <span class="session-time">{ago}</span>
        </div>"""
    return rows


def show():
    init_session_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── GUARD: Model Must Be Ready ─────────────────────────────────────────
    if not is_model_ready():
        st.markdown("""
            <div style="background:#fff8f8;border:1px solid rgba(186,26,26,0.2);border-radius:10px;
                        padding:20px 24px;display:flex;align-items:center;gap:12px;">
              <span class="material-symbols-outlined" style="color:#ba1a1a;font-size:24px;">warning</span>
              <div>
                <p style="font-weight:700;color:#ba1a1a;margin:0;font-size:14px;font-family:Inter,sans-serif;">
                  Model Not Trained</p>
                <p style="color:#64748b;margin:4px 0 0;font-size:13px;font-family:Inter,sans-serif;">
                  Register users and train the model before attempting authentication.</p>
              </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # ── Load recent sessions ───────────────────────────────────────────────
    recent_events = get_auth_events(limit=3)

    # ── Page Header ───────────────────────────────────────────────────────
    st.markdown("""
        <h1 class="login-title">Verify Your Identity</h1>
        <div class="login-title-bar"></div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────
    col_cam, col_panel = st.columns([8, 4], gap="large")

    with col_cam:
        # Browser-native camera input (Cloud Compatible)
        captured_file = st.camera_input("Capture Face for Authentication")

        confidence = "--"
        status_text = "Ready to Scan"

        if captured_file:
            # ── Process Capture (Only once per click) ──────────────────
            with st.spinner("Analyzing biometric signature..."):
                frame = _file_to_cv2(captured_file)
                result = predict_face(frame)
                
                # Logic to handle results (saved in session state for persistence)
                if result["status"] == "authenticated":
                    dist = result.get("distance") or 0.0
                    conf_val = max(0.0, (1 - dist/THRESHOLD) * 100)
                    confidence = f"{int(conf_val)}"
                    status_text = "AUTHENTICATED"
                    log_auth_event("authenticated", result["username"], conf_val)
                    st.session_state.auth_result = {
                        "success": True, 
                        "username": result["username"],
                        "confidence": conf_val
                    }
                else:
                    status_text = "FACE NOT RECOGNIZED"
                    log_auth_event("unknown", None, None)
                    st.session_state.auth_result = {"success": False}
                
                # Update login attempt counter
                from app.utils import increment_login_attempts
                increment_login_attempts()
                # We don't rerun immediately so user can see the result in the 'Status' card below
        else:
            # Check if there's a persistent result in session state
            res = st.session_state.get("auth_result")
            if res:
                if res.get("success"):
                    confidence = f"{int(res.get('confidence', 0))}"
                    status_text = "AUTHENTICATED"
                else:
                    status_text = "FACE NOT RECOGNIZED"

        # ── Persistent Auth Result Message ────────────────────────────
        res = st.session_state.get("auth_result")
        if res:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if res["success"]:
                st.markdown(f"""
                    <div style="background:#f0fdf9;border:1px solid rgba(0,132,114,0.25);border-radius:10px;
                                padding:18px 24px;display:flex;align-items:center;gap:14px;">
                      <span class="material-symbols-outlined"
                            style="color:#008472;font-size:28px;
                                   font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24;">
                        check_circle
                      </span>
                      <div>
                        <p style="font-weight:700;color:#008472;margin:0;font-size:15px;font-family:Inter,sans-serif;">
                          Authentication Successful</p>
                        <p style="color:#64748b;margin:4px 0 0;font-size:13px;font-family:Inter,sans-serif;">
                          Welcome back, <strong>{res.get('username', 'User')}</strong></p>
                      </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background:#fff8f8;border:1px solid rgba(186,26,26,0.2);border-radius:10px;
                                padding:18px 24px;display:flex;align-items:center;gap:14px;">
                      <span class="material-symbols-outlined" style="color:#ba1a1a;font-size:28px;">cancel</span>
                      <div>
                        <p style="font-weight:700;color:#ba1a1a;margin:0;font-size:15px;font-family:Inter,sans-serif;">
                          Authentication Failed</p>
                        <p style="color:#64748b;margin:4px 0 0;font-size:13px;font-family:Inter,sans-serif;">
                          Face not recognized. Please adjust lighting and try again.</p>
                      </div>
                    </div>
                """, unsafe_allow_html=True)

        # ── Stat micro-cards ──────────────────────────────────────────
        st.markdown(f"""
            <div class="stat-cards">
              <div class="stat-card" style="border-left:3px solid #b81120;">
                <span class="stat-card-label" style="color:#b81120;">Liveness Score</span>
                <span class="stat-card-value">{confidence}<span class="stat-card-unit">/ 100</span></span>
              </div>
              <div class="stat-card" style="border-left:3px solid #0060ab;">
                <span class="stat-card-label" style="color:#0060ab;">Encryption</span>
                <span class="stat-card-value" style="font-size:1rem;">AES-256
                  <span class="material-symbols-outlined"
                        style="font-size:14px;color:#0060ab;vertical-align:middle;
                               font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24;">
                    lock
                  </span>
                </span>
              </div>
              <div class="stat-card" style="border-left:3px solid #008472;">
                <span class="stat-card-label" style="color:#008472;">Status</span>
                <span class="stat-card-value" style="font-size:0.8rem;color:#64748b;">{status_text}</span>
              </div>
            </div>
        """, unsafe_allow_html=True)

    with col_panel:
        # ── Auth info panel ───────────────────────────────────────────
        last_user = ""
        if recent_events and recent_events[0].get("status") == "authenticated":
            last_user = recent_events[0].get("username", "")
        user_display = last_user.upper().replace(" ", "_") if last_user else "—"

        st.markdown(f"""
            <div class="auth-panel">
              <div class="auth-panel-title">
                <span class="material-symbols-outlined auth-panel-title-icon"
                      style="font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24;">
                  verified_user
                </span>
                Identity Hub
              </div>
              <span class="auth-field-label">Last Verified User</span>
              <div class="auth-field-value">{user_display}</div>
              <p class="auth-hint">
                Position your face within the camera frame above and click the capture button.
              </p>
              <div class="auth-info-chip">
                <span class="material-symbols-outlined auth-info-chip-icon">info</span>
                <span class="auth-info-chip-text">Cloud Mode Active</span>
              </div>
            </div>
        """, unsafe_allow_html=True)

        # Recent sessions
        sessions_html = _recent_sessions_html(recent_events)
        st.markdown(f"""
            <div class="sessions-card">
              <span class="sessions-label">Recent Sessions</span>
              {sessions_html}
            </div>
        """, unsafe_allow_html=True)
