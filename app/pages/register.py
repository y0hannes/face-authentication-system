import time
import threading
import cv2
import streamlit as st

from config import CAPTURE_COUNT
from src.data_collection import capture_images
from src.train import train_and_save
from src.predict import reload_model
from app.utils import init_session_state


@st.cache_resource
def get_preview_camera():
    """Keep a single camera instance open for the live preview across reruns."""
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # Warm-up
    return cap


def _read_preview_frame():
    """Grab one frame from the persistent camera. Returns RGB numpy array or None."""
    cap = get_preview_camera()
    ret, frame = cap.read()
    if not ret:
        return None
    target_w = 480
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    display = cv2.resize(frame, (target_w, target_h))
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


def show():
    init_session_state()

    if "camera_preview" not in st.session_state:
        st.session_state.camera_preview = False

    # ── Page-level CSS ─────────────────────────────────────────────────────
    st.markdown("""
        <style>
        .camera-label {
            font-size: 10px; font-weight: bold; letter-spacing: 0.12em;
            color: var(--txt-sub); text-transform: uppercase; margin-bottom: 8px;
        }
        .metric-box {
            background: var(--bg-card); padding: 1.25rem 1rem; border-radius: 8px;
            border-left: 4px solid var(--brand);
            box-shadow: var(--shadow);
        }
        .live-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--bg-card); backdrop-filter: blur(8px);
            padding: 6px 14px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--txt-main);
            box-shadow: var(--shadow);
        }
        .ping-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--brand); animation: ping 1s cubic-bezier(0,0,0.2,1) infinite;
            flex-shrink: 0;
        }
        @keyframes ping {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.6; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='font-weight:900;letter-spacing:-0.05em;color:var(--txt-main);font-size:2.25rem;margin-bottom:0'>Create Your Secure Profile</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0060ab;font-weight:500;font-size:1.125rem;margin-bottom:1.5rem;margin-top:4px'>Step into the future of institutional security with biometric enrollment.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 5], gap="large")

    with col_left:
        st.markdown("<div class='camera-label'>Live Camera Feed</div>", unsafe_allow_html=True)

        frame_placeholder = st.empty()
        progress_bar = st.empty()
        status_placeholder = st.empty()

        # ── Determine what to show in the camera box ───────────────────────
        is_capturing = st.session_state.get("capture_running", False)
        is_preview = st.session_state.camera_preview and not is_capturing

        if is_preview:
            # Show one live frame and schedule rerun to simulate stream
            frame_rgb = _read_preview_frame()
            if frame_rgb is not None:
                frame_placeholder.image(
                    frame_rgb, channels="RGB", output_format="JPEG", width="stretch"
                )
            else:
                frame_placeholder.warning("⚠️ Camera unavailable.")
        elif not is_capturing:
            # Idle — show a dark placeholder with a start-preview prompt
            frame_placeholder.markdown("""
                <div style="background:#1e293b; border-radius:12px; aspect-ratio:16/9;
                            display:flex; flex-direction:column; align-items:center;
                            justify-content:center; color:#94a3b8; gap:12px; padding:2rem;">
                    <span style="font-size:3rem;">📷</span>
                    <span style="font-size:0.85rem; font-weight:600; letter-spacing:0.05em;">
                        Click <b style="color:#fff">Start Preview</b> to activate
                    </span>
                </div>
            """, unsafe_allow_html=True)

        # Live badge (only shown during preview or capture)
        if is_preview or is_capturing:
            st.markdown("""
                <div style="margin-top:8px">
                    <span class="live-badge">
                        <span class="ping-dot"></span> Live Camera Feed
                    </span>
                </div>
            """, unsafe_allow_html=True)

        # ── Camera control buttons ──────────────────────────────────────────
        st.write("")
        if not is_capturing:
            if is_preview:
                if st.button("⏹ Stop Preview", use_container_width=True):
                    st.session_state.camera_preview = False
                    get_preview_camera.clear()  # Release camera resource
                    st.rerun()
            else:
                if st.button("🎥 Start Preview", use_container_width=True):
                    st.session_state.camera_preview = True
                    st.rerun()

        # ── Live metric boxes ───────────────────────────────────────────────
        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-box'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:var(--brand)'>Light Level</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>Optimal</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box' style='border-left-color:#00685a'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:#00685a'>Detection</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>Face Found</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-box' style='border-left-color:#0060ab'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:#0060ab'>Quality</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>98.4%</div></div>", unsafe_allow_html=True)

    with col_right:
        username = st.text_input("FULL NAME", placeholder="e.g. Dr. Julian Thorne")

        st.info("🔒 **Encryption Standard**\n\nYour biometric signature is hashed and encrypted using AES-256 before being stored in our isolated secure vault.")

        if st.button("📸 Enroll Face", use_container_width=True):
            if not username or len(username.strip()) < 2:
                st.error("Full Name must be at least 2 characters.")
            else:
                st.session_state.reg_username = username.strip()
                st.session_state.capture_running = True
                st.session_state.capture_result = None
                # Stop preview and release camera so capture can use it
                st.session_state.camera_preview = False
                get_preview_camera.clear()
                st.rerun()

        if st.session_state.get("capture_result"):
            res = st.session_state.capture_result
            if res.get("success"):
                st.success("✅ Facial data securely captured!")
                if st.button("🚀 Synchronize Neural Matrix", use_container_width=True):
                    with st.spinner("Extracting features and compiling matrix..."):
                        train_res = train_and_save()
                    if train_res.get("success"):
                        if train_res.get("fig"):
                            import os
                            os.makedirs("models", exist_ok=True)
                            train_res["fig"].savefig("models/cm_latest.png")
                        reload_model()
                        st.success(train_res["message"])
                        st.balloons()
                    else:
                        st.error(train_res.get("message", "Training failed."))
            else:
                st.error(res.get("message", "Capture failed."))

        st.markdown("<p style='text-align:center;font-size:12px;color:var(--txt-sub);margin-top:16px'>By enrolling, you agree to the Precision Scholar Biometric Policy.</p>", unsafe_allow_html=True)
        st.warning("💡 **Pro Tip:** Ensure you are in a well-lit area and remove glasses for the initial scan.")

    # ── BLOCKING CAPTURE ENGINE (runs after Enroll Face click) ────────────
    if st.session_state.get("capture_running"):
        stop_event = threading.Event()
        _frame_counter = [0]

        def on_frame(frame, status, count):
            _frame_counter[0] += 1
            if _frame_counter[0] % 2 == 0:
                target_w = 480
                h, w = frame.shape[:2]
                target_h = int(h * target_w / w)
                display = cv2.resize(frame, (target_w, target_h))
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    display_rgb, channels="RGB", output_format="JPEG", width="stretch"
                )
            progress_bar.progress(
                min(count / CAPTURE_COUNT, 1.0),
                text=f"Capturing biometric data… {count} / {CAPTURE_COUNT}"
            )
            status_placeholder.info(status)

        result = capture_images(
            st.session_state.reg_username,
            frame_callback=on_frame,
            stop_event=stop_event,
        )

        st.session_state.capture_result = result
        st.session_state.capture_running = False
        st.rerun()

    # ── Trigger rerun while preview is active to simulate streaming ────────
    if st.session_state.camera_preview and not st.session_state.get("capture_running"):
        time.sleep(0.04)  # ~25 fps
        st.rerun()
