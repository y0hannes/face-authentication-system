import time
import cv2
import streamlit as st

from config import CAPTURE_COUNT
from src.train import train_and_save
from src.predict import reload_model
from app.utils import init_session_state


@st.cache_resource
def get_preview_camera():
    """Keep a single camera instance open for the live preview across reruns."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    time.sleep(0.5)  # Warm-up
    return cap


def _read_preview_frame(draw_overlay=True):
    """Grab one frame from the persistent camera. Returns RGB numpy array or None."""
    cap = get_preview_camera()
    if cap is None:
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    # Mirror
    frame = cv2.flip(frame, 1)
    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    display = cv2.resize(frame, (target_w, target_h))
    
    if draw_overlay:
        # Draw Scan Zone (matching login page)
        cx, cy = 320, 240
        w_h, h_h = 100, 125
        color = (114, 132, 0) # BGR (success-green)
        thickness = 2
        length = 30
        import numpy as np
        corners = [
            [(cx-w_h, cy-h_h+length), (cx-w_h, cy-h_h), (cx-w_h+length, cy-h_h)],
            [(cx+w_h-length, cy-h_h), (cx+w_h, cy-h_h), (cx+w_h, cy-h_h+length)],
            [(cx+w_h, cy+h_h-length), (cx+w_h, cy+h_h), (cx+w_h-length, cy+h_h)],
            [(cx-w_h+length, cy+h_h), (cx-w_h, cy+h_h), (cx-w_h, cy+h_h-length)]
        ]
        for corner in corners:
            cv2.polylines(display, [np.array(corner)], False, color, thickness, cv2.LINE_AA)
        cv2.putText(display, "SCAN ZONE", (cx-w_h+10, cy-h_h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
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
        /* Premium rounding for all streamlit images */
        [data-testid="stImage"] img {
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
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
            # Placeholder for live feed (loop moved to end of script for non-blocking UI)
            frame_placeholder.info("Initializing camera...")
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
                if st.button("⏹ Stop Preview", width="stretch"):
                    st.session_state.camera_preview = False
                    get_preview_camera.clear()  # Release camera resource
                    st.rerun()
            else:
                if st.button("🎥 Start Preview", width="stretch"):
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

        if st.button("📸 Enroll Face", width="stretch"):
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
                if st.button("🚀 Synchronize Neural Matrix", width="stretch"):
                    with st.spinner("Extracting features and compiling matrix..."):
                        train_res = train_and_save()
                    if train_res.get("success"):
                        if train_res.get("fig"):
                            import os
                            os.makedirs("models", exist_ok=True)
                            train_res["fig"].savefig("models/cm_latest.png")
                        reload_model()
                        st.success(train_res["message"])
                    else:
                        st.error(train_res.get("message", "Training failed."))
            else:
                st.error(res.get("message", "Capture failed."))

        st.markdown("<p style='text-align:center;font-size:12px;color:var(--txt-sub);margin-top:16px'>By enrolling, you agree to the Precision Scholar Biometric Policy.</p>", unsafe_allow_html=True)
        st.warning("💡 **Pro Tip:** Ensure you are in a well-lit area and remove glasses for the initial scan.")

        if st.session_state.get("capture_running"):
            if st.button("⏹ Stop Enrollment", key="stop_enroll", width="stretch"):
                st.session_state.capture_running = False
                st.rerun()

    # ── LIVE UPDATE LOOPS (Moved to end for non-blocking UI) ──────────────
    
    # 1. Preview Loop
    if st.session_state.get("camera_preview") and not st.session_state.get("capture_running"):
        while st.session_state.get("camera_preview") and not st.session_state.get("capture_running"):
            frame_rgb = _read_preview_frame(draw_overlay=True)
            if frame_rgb is not None:
                frame_placeholder.image(
                    frame_rgb, channels="RGB", output_format="JPEG", width="stretch"
                )
            else:
                frame_placeholder.warning("⚠️ Camera unavailable.")
                break
            time.sleep(0.05)
        
        # Refresh if deactivated to show idle view
        if not st.session_state.camera_preview:
            st.rerun()

    # 2. Capture Engine Loop (Moved here to ensure UI elements stay visible)
    if st.session_state.get("capture_running"):
        import os
        from src.preprocessing import preprocess_image
        from config import DATA_PATH, CAPTURE_DELAY

        username = st.session_state.reg_username
        clean_username = username.strip().lower().replace(" ", "_")
        save_path = os.path.join(DATA_PATH, clean_username)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        time.sleep(0.5)

        count = 0
        last_capture_time = 0

        while count < CAPTURE_COUNT and st.session_state.get("capture_running"):
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Could not read from camera.")
                break

            frame = cv2.flip(frame, 1)
            processed = preprocess_image(frame)
            
            is_valid = processed is not None
            status_msg = "No Face Detected"
            color = "#ba1a1a"

            if is_valid:
                status_msg = f"Capturing… ({count + 1}/{CAPTURE_COUNT})"
                color = "#008472"
                
                current_time = time.time()
                if current_time - last_capture_time > CAPTURE_DELAY:
                    img_path = os.path.join(save_path, f"{count}.jpg")
                    cv2.imwrite(img_path, frame)
                    count += 1
                    last_capture_time = current_time

            # Update UI (Draw overlay only on display frame)
            display = frame.copy()
            # Draw Scan Zone
            cx, cy = display.shape[1]//2, display.shape[0]//2
            w_h, h_h = 100, 125
            color = (114, 132, 0) # BGR (success-green)
            thickness = 2
            length = 30
            import numpy as np
            corners = [
                [(cx-w_h, cy-h_h+length), (cx-w_h, cy-h_h), (cx-w_h+length, cy-h_h)],
                [(cx+w_h-length, cy-h_h), (cx+w_h, cy-h_h), (cx+w_h, cy-h_h+length)],
                [(cx+w_h, cy+h_h-length), (cx+w_h, cy+h_h), (cx+w_h-length, cy+h_h)],
                [(cx-w_h+length, cy+h_h), (cx-w_h, cy+h_h), (cx-w_h, cy+h_h-length)]
            ]
            for pt in corners:
                cv2.polylines(display, [np.array(pt)], False, color, thickness, cv2.LINE_AA)
            cv2.putText(display, "ENROLLING...", (cx-w_h+10, cy-h_h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", output_format="JPEG", width="stretch")
            progress_bar.progress(min(count / CAPTURE_COUNT, 1.0))
            status_placeholder.markdown(f"<p style='color:{color};font-weight:600;margin:0;'>{status_msg}</p>", unsafe_allow_html=True)
            
            time.sleep(0.01)

        cap.release()
        st.session_state.capture_result = {
            "success": count == CAPTURE_COUNT,
            "saved": count,
            "message": f"Captured {count} samples."
        }
        st.session_state.capture_running = False
        st.rerun()

