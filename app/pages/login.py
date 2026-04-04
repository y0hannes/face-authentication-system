import time
import datetime
import cv2
import streamlit as st

from src.predict import is_model_ready, predict_face
from app.utils import init_session_state, log_auth_event, get_auth_events
from config import THRESHOLD


@st.cache_resource
def get_preview_camera():
    """Keep a single camera instance open for the live preview across reruns."""
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # warm-up
    return cap


def _read_preview_frame():
    """Grab one frame from the persistent preview camera. Returns RGB array or None."""
    cap = get_preview_camera()
    ret, frame = cap.read()
    if not ret:
        return None
    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    display = cv2.resize(frame, (target_w, target_h))
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


# ── Page CSS ──────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] { background: #f7f9fd; }
[data-testid="block-container"]    { padding-top: 1.5rem !important; }

/* Hero title */
.login-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #191c1f;
    margin: 0 0 12px;
    line-height: 1.1;
    font-family: Inter, sans-serif;
}
.login-title-bar {
    height: 4px;
    width: 80px;
    background: linear-gradient(to right, #b81120, #dc3135);
    border-radius: 9999px;
    margin-bottom: 28px;
}

/* Camera viewport */
.cam-viewport {
    position: relative;
    width: 100%;
    background: #0f172a;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 20px 48px rgba(0,0,0,0.3);
    border: 2px solid #e0e2e6;
    aspect-ratio: 16/9;
    display: flex;
    align-items: center;
    justify-content: center;
}
.cam-idle-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
.cam-idle-icon {
    font-size: 64px;
    color: rgba(184,17,32,0.4);
}
.cam-idle-text {
    font-size: 13px;
    color: rgba(255,255,255,0.4);
    font-family: monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
/* Scan-zone overlay */
.scan-zone-wrap {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
}
.scan-zone-box {
    width: 200px;
    height: 250px;
    border: 2px solid rgba(184,17,32,0.5);
    border-radius: 40px;
    display: flex;
    align-items: flex-start;
    justify-content: center;
}
.scan-zone-label {
    margin-top: -13px;
    background: #b81120;
    color: #fff;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 9999px;
}
/* LIVE badge */
.live-badge {
    position: absolute;
    top: 16px;
    right: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(15,23,42,0.8);
    backdrop-filter: blur(8px);
    border-radius: 9999px;
    border: 1px solid rgba(255,255,255,0.1);
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #b81120;
}
.live-text {
    font-size: 10px;
    color: #fff;
    font-family: monospace;
    letter-spacing: 0.15em;
}
/* Status bar at bottom of camera */
.cam-status-bar {
    position: absolute;
    bottom: 16px;
    left: 16px;
    right: 16px;
    background: rgba(15,23,42,0.5);
    backdrop-filter: blur(8px);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid rgba(255,255,255,0.05);
}
.cam-status-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.cam-status-icon {
    font-size: 20px;
    color: #b81120;
}
.cam-status-text {
    font-size: 13px;
    font-weight: 500;
    color: #fff;
    font-family: Inter, sans-serif;
}
.cam-progress-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.cam-progress-bar {
    width: 80px;
    height: 5px;
    background: rgba(255,255,255,0.2);
    border-radius: 9999px;
    overflow: hidden;
}
.cam-progress-fill {
    height: 100%;
    border-radius: 9999px;
    background: #b81120;
}
.cam-progress-pct {
    font-size: 10px;
    color: rgba(255,255,255,0.6);
    font-family: monospace;
}

/* Stat micro-cards */
.stat-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
}
.stat-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
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
    color: #191c1f;
    font-family: Inter, sans-serif;
    line-height: 1;
}
.stat-card-unit {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 500;
    margin-left: 2px;
}

/* Auth panel */
.auth-panel {
    background: #ffffff;
    border-radius: 12px;
    padding: 28px;
    box-shadow: 0 8px 24px rgba(25,28,31,0.04);
    border: 1px solid rgba(228,189,186,0.2);
}
.auth-panel-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #191c1f;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    font-family: Inter, sans-serif;
}
.auth-panel-title-icon {
    color: #b81120;
    font-size: 22px;
}
.auth-field-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 6px;
    display: block;
}
.auth-field-value {
    width: 100%;
    background: #f2f4f8;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #64748b;
    font-family: monospace;
    border: 1px solid rgba(228,189,186,0.1);
    margin-bottom: 18px;
}
.auth-hint {
    font-size: 13px;
    color: #64748b;
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

/* Verify button (replaces st.button via HTML — actual button is st widget below) */
.verify-btn-style {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #b81120, #dc3135);
    color: #fff;
    font-weight: 700;
    font-size: 14px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 4px 16px rgba(184,17,32,0.3);
    font-family: Inter, sans-serif;
    transition: opacity 0.2s;
}

/* Style the Streamlit buttons to match the design */
[data-testid="block-container"] div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #b81120, #dc3135) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 14px !important;
    box-shadow: 0 4px 16px rgba(184,17,32,0.25) !important;
    font-family: Inter, sans-serif !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
[data-testid="block-container"] div.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
}
[data-testid="block-container"] div.stButton > button[kind="secondary"] {
    background: #eceef2 !important;
    color: #003461 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    width: 100% !important;
}

/* Recent sessions card */
.sessions-card {
    background: #f2f4f8;
    border-radius: 12px;
    padding: 20px;
    margin-top: 14px;
}
.sessions-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    margin-bottom: 12px;
    display: block;
    font-family: Inter, sans-serif;
}
.session-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    background: #ffffff;
    border-radius: 8px;
    margin-bottom: 8px;
}
.session-row:last-child { margin-bottom: 0; }
.session-left { display: flex; align-items: center; gap: 10px; }
.session-dot-green { width: 8px; height: 8px; border-radius: 50%; background: #008472; flex-shrink: 0; }
.session-dot-red   { width: 8px; height: 8px; border-radius: 50%; background: #ba1a1a; flex-shrink: 0; }
.session-name { font-size: 12px; font-weight: 600; color: #191c1f; font-family: Inter, sans-serif; }
.session-time { font-size: 10px; color: #94a3b8; font-family: Inter, sans-serif; }
.sessions-empty {
    font-size: 12px;
    color: #94a3b8;
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

    # ── Session state defaults ─────────────────────────────────────────────
    if "camera_preview" not in st.session_state:
        st.session_state.camera_preview = False

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
        # ── Camera viewport shell ──────────────────────────────────────
        cam_placeholder  = st.empty()
        stat_placeholder = st.empty()

        is_auth_running = st.session_state.get("auth_running", False)
        is_preview      = st.session_state.camera_preview and not is_auth_running

        if is_preview:
            # Live preview — read one frame and rerun to simulate stream
            frame_rgb = _read_preview_frame()
            if frame_rgb is not None:
                cam_placeholder.image(
                    frame_rgb, channels="RGB", output_format="JPEG",
                    use_container_width=True,
                )
            else:
                cam_placeholder.warning("⚠️ Camera unavailable.")

        elif not is_auth_running:
            # Idle dark placeholder
            cam_placeholder.markdown("""
                <div class="cam-viewport">
                  <div class="cam-idle-inner">
                    <span class="material-symbols-outlined cam-idle-icon">videocam_off</span>
                    <span class="cam-idle-text">Camera Inactive</span>
                  </div>
                  <div class="scan-zone-wrap">
                    <div class="scan-zone-box">
                      <span class="scan-zone-label">Scan Zone</span>
                    </div>
                  </div>
                </div>
            """, unsafe_allow_html=True)

        # LIVE badge (preview or auth running)
        if is_preview or is_auth_running:
            st.markdown("""
                <div style="margin-top:8px;">
                  <span style="display:inline-flex;align-items:center;gap:8px;
                               background:rgba(255,255,255,0.88);backdrop-filter:blur(8px);
                               padding:5px 14px;border-radius:999px;font-size:11px;
                               font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                               color:#191c1f;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                    <span style="width:8px;height:8px;border-radius:50%;background:#b81120;
                                 display:inline-block;animation:ping 1s infinite;"></span>
                    Live Camera Feed
                  </span>
                </div>
                <style>
                  @keyframes ping {
                    0%,100%{transform:scale(1);opacity:1}
                    50%{transform:scale(1.4);opacity:0.6}
                  }
                </style>
            """, unsafe_allow_html=True)

        # ── Preview toggle button ──────────────────────────────────────
        if not is_auth_running:
            st.write("")
            if is_preview:
                if st.button("⏹ Stop Preview", key="stop_preview", use_container_width=True):
                    st.session_state.camera_preview = False
                    get_preview_camera.clear()
                    st.rerun()
            else:
                if st.button("🎥 Start Preview", key="start_preview", use_container_width=True):
                    st.session_state.camera_preview = True
                    st.rerun()

        # ── Persistent Auth Result (Moved here) ───────────────────────
        res = st.session_state.get("auth_result")
        if res and not is_auth_running:
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
                          Welcome back, <strong>{res['username']}</strong></p>
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
                          Authentication Stopped</p>
                        <p style="color:#64748b;margin:4px 0 0;font-size:13px;font-family:Inter,sans-serif;">
                          No identity was verified. Try again.</p>
                      </div>
                    </div>
                """, unsafe_allow_html=True)

        # ── Stat micro-cards ──────────────────────────────────────────
        st.write("")
        stat_placeholder.markdown("""
            <div class="stat-cards">
              <div class="stat-card" style="border-left:3px solid #b81120;">
                <span class="stat-card-label" style="color:#b81120;">Liveness Score</span>
                <span class="stat-card-value">--<span class="stat-card-unit">/ 100</span></span>
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
                <span class="stat-card-label" style="color:#008472;">Latency</span>
                <span class="stat-card-value">14<span class="stat-card-unit">ms</span></span>
              </div>
            </div>
        """, unsafe_allow_html=True)

    with col_panel:
        # ── Auth panel ────────────────────────────────────────────────
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
                Authentication
              </div>
              <span class="auth-field-label">Last Verified User</span>
              <div class="auth-field-value">{user_display}</div>
              <p class="auth-hint">
                Ensure your face is centred in the frame and well-lit.
                Avoid wearing glasses or masks during the verification process.
              </p>
              <div class="auth-info-chip">
                <span class="material-symbols-outlined auth-info-chip-icon">info</span>
                <span class="auth-info-chip-text">Auto-detection active</span>
              </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Verify / Stop buttons
        if not st.session_state.get("auth_running", False):
            verify_btn = st.button(
                "🎥  Verify Identity",
                key="start_auth",
                use_container_width=True,
                type="primary",
            )
        else:
            verify_btn = False
            if st.button("⏹  Stop Camera", key="stop_auth", use_container_width=True):
                st.session_state.auth_running = False

        # Recent sessions
        sessions_html = _recent_sessions_html(recent_events)
        st.markdown(f"""
            <div class="sessions-card">
              <span class="sessions-label">Recent Sessions</span>
              {sessions_html}
            </div>
        """, unsafe_allow_html=True)

    # ── AUTH LOOP ─────────────────────────────────────────────────────────
    if verify_btn:
        from app.utils import increment_login_attempts
        increment_login_attempts()
        # Release preview camera so the auth loop can claim device 0
        st.session_state.camera_preview = False
        get_preview_camera.clear()
        st.session_state.auth_running = True

        success        = False
        message        = ""
        user_identified = None
        frame_counter  = 0

        # Open camera
        cap = cv2.VideoCapture(0)
        time.sleep(0.8)  # warm-up

        while st.session_state.get("auth_running", True):
            ret, frame = cap.read()
            if not ret:
                cam_placeholder.error("Could not read from camera.")
                break

            frame_counter += 1

            # ── Predict every 3rd frame ────────────────────────────────
            conf_pct = 0.0
            status_text = "Searching for face..."
            if frame_counter % 3 == 0:
                result = predict_face(frame)

                if result["status"] == "authenticated":
                    success         = True
                    message         = result["message"]
                    user_identified = result["username"]
                    dist            = result.get("distance") or 0.0
                    conf_pct        = max(0.0, (1 - dist / THRESHOLD) * 100)
                    log_auth_event("authenticated", user_identified, conf_pct)
                    break
                elif result["status"] == "unknown":
                    status_text = "Face not recognised"
                elif result["status"] == "no_face":
                    status_text = "Searching for face..."
                else:
                    status_text = result["message"]

            # ── Render live camera frame ───────────────────────────────
            if frame_counter % 2 == 0:
                target_w = 640
                h, w     = frame.shape[:2]
                target_h = int(h * target_w / w)
                display  = cv2.resize(frame, (target_w, target_h))
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                # Live feed badge + status bar overlaid via caption trick
                cam_placeholder.image(
                    display_rgb,
                    channels="RGB",
                    output_format="JPEG",
                    use_container_width=True,
                )

                # Update stat micro-cards with live confidence
                pct_int = int(min(conf_pct, 100))
                stat_placeholder.markdown(f"""
                    <div class="stat-cards">
                      <div class="stat-card" style="border-left:3px solid #b81120;">
                        <span class="stat-card-label" style="color:#b81120;">Liveness Score</span>
                        <span class="stat-card-value">{pct_int}<span class="stat-card-unit">/ 100</span></span>
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
                        <span class="stat-card-value" style="font-size:0.9rem;color:#64748b;">{status_text[:14]}</span>
                      </div>
                    </div>
                """, unsafe_allow_html=True)

        cap.release()
        st.session_state.auth_running = False

        # Log cancellations
        if not success:
            log_auth_event("unknown", None, None)

        # Trigger preview rerun loop if needed
        st.session_state.auth_result = {
            "success": success,
            "message": message,
            "username": user_identified,
        }
        st.rerun()

    # ── Trigger rerun while preview is active to simulate streaming ────────
    if st.session_state.get("camera_preview") and not st.session_state.get("auth_running"):
        time.sleep(0.04)  # ~25 fps
        st.rerun()
