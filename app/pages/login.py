import time
import cv2
import streamlit as st

from src.predict import is_model_ready, predict_face
from app.utils import init_session_state


def show():
    init_session_state()

    # ── GUARD: Model Must Be Ready ─────────────────────────────────────────
    if not is_model_ready():
        st.warning("⚠️ The model is not trained yet. Please register users and train the model first.")
        return

    # ── SETUP UI ───────────────────────────────────────────────────────────
    main_ui = st.empty()

    with main_ui.container():
        st.title("Login — Face Authentication")
        st.markdown(
            "Start the camera to authenticate. Position your face in the frame. "
            "The system will automatically log you in if it recognizes you."
        )
        st.divider()

        col1, col2 = st.columns([1, 1])
        start_btn = col1.button("▶ Start Authentication", width="stretch", type="primary")
        stop_btn  = col2.button("⏹ Stop Camera",          width="stretch")

        if stop_btn:
            st.session_state.auth_running = False

    # ── AUTHENTICATION UI (FOCUS MODE) ─────────────────────────────────────
    if start_btn:
        main_ui.empty()
        st.session_state.auth_running = True
        
        camera_ui = st.empty()
        
        with camera_ui.container():
            st.header("Scanning Face...")
            st.button("⏹ Cancel", key="cancel_auth_btn_temp")
            
            col_left, col_cam, col_right = st.columns([1, 4, 1])
            with col_cam:
                status_placeholder = st.empty()
                frame_placeholder  = st.empty()

        # Start Camera
        cap = cv2.VideoCapture(0)
        
        # Warm up camera
        time.sleep(1.0)
        
        success = False
        message = ""
        user_identified = None

        frame_counter = 0

        while st.session_state.auth_running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Could not read from camera.")
                break
                
            frame_counter += 1

            # Only run heavy face prediction every 3rd frame to keep UI responsive
            if frame_counter % 3 == 0:
                result = predict_face(frame)
                
                status_text = result["message"]
                
                # Check outcome
                if result["status"] == "authenticated":
                    success = True
                    message = result["message"]
                    user_identified = result["username"]
                    break  # Exit loop immediately on successful auth
                elif result["status"] == "unknown":
                    status_placeholder.warning(f"🤷 {status_text}")
                elif result["status"] == "no_face":
                    status_placeholder.info("⏳ Waiting for face...")
                else:
                    status_placeholder.error(status_text)

            # Update display every other frame
            if frame_counter % 2 == 0:
                target_w = 480
                h, w = frame.shape[:2]
                target_h = int(h * target_w / w)
                display = cv2.resize(frame, (target_w, target_h))
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    display_rgb,
                    channels="RGB",
                    output_format="JPEG",
                    width="stretch"
                )

        # Cleanup
        cap.release()
        
        # Force the session state to False so we don't keep looping
        st.session_state.auth_running = False
        
        # Save results to session state
        st.session_state.auth_result = {
            "success": success,
            "message": message,
            "username": user_identified
        }
        
        # Rerun to restore UI
        st.rerun()

    # ── PERSISTENT RESULT ──────────────────────────────────────────────────
    if not st.session_state.get("auth_running", False) and st.session_state.get("auth_result"):
        res = st.session_state.auth_result
        if res["success"]:
            st.success(f"🎊 {res['message']}")
            st.balloons()
            st.info(f"You are securely logged in as **{res['username']}**.")
        else:
            # If they canceled explicitly
            st.info("Authentication stopped.")

