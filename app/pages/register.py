import threading
import cv2
import streamlit as st

from config import CAPTURE_COUNT
from src.data_collection import capture_images
from src.train import train_and_save
from src.predict import reload_model
from app.utils import init_session_state, user_image_count


def show():
    init_session_state()

    # ── SETUP UI ───────────────────────────────────────────────────────────
    # We wrap the standard UI in a placeholder. During capture, we can hide it.
    main_ui = st.empty()
    
    with main_ui.container():
        st.title("Register New User")
        st.markdown("Capture your face images, then train the model.")
        st.divider()

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
                    f"**'{username}'** already has **{existing_count}** image(s). "
                    "Capturing again will add more. Retrain the model afterwards."
                )

        st.divider()

        if not username or len(username.strip()) < 2:
            st.info("Enter a valid username above to begin capturing.")
            return

        st.subheader("📸 Face Capture")
        st.markdown(
            f"Stand in front of your camera. **{CAPTURE_COUNT} images** will be "
            "captured automatically when your face is clearly visible."
        )

        col1, col2 = st.columns([1, 1])
        start_btn = col1.button("▶ Start Capture", width="stretch", type="primary")
        stop_btn  = col2.button("⏹ Stop",          width="stretch")

        if stop_btn and "stop_event" in st.session_state:
            st.session_state.stop_event.set()
            st.session_state.capture_running = False

        st.divider()

        # ── Train Model Section ────────────────────────────────────────────────
        st.subheader("🧠 Train Model")
        st.markdown(
            "After capturing your images, click below to train the facial recognition "
            "model. **You must train the model before the Login page will unlock.**"
        )

        train_btn = st.button("🚀 Train Model", width="stretch")
        train_status = st.empty()
        train_progress = st.empty()

        if train_btn:
            train_status.info("Starting training process...")

            def _on_train_progress(current: int, total: int):
                train_progress.text(f"Processing user {current} of {total}...")

            with st.spinner("Extracting features and training the model..."):
                train_result = train_and_save(progress_callback=_on_train_progress)

            train_progress.empty()
            
            if train_result.get("success"):
                reload_model()  # Crucial: wipe the old cached model from memory!
                train_status.success(train_result["message"])
                st.balloons()
                
                # Show the confusion matrix
                if train_result.get("fig"):
                    st.pyplot(train_result["fig"])
            else:
                train_status.error(train_result.get("message", "Training failed."))

    # ── CAPTURE UI (FOCUS MODE) ────────────────────────────────────────────
    # This block triggers when "Start Capture" is clicked.
    if start_btn:
        # Hide the setup form entirely! The camera jumps perfectly to the top center.
        main_ui.empty()

        st.session_state.capture_running = True
        st.session_state.capture_result  = None
        stop_event = threading.Event()
        st.session_state.stop_event = stop_event

        camera_ui = st.empty()
        
        with camera_ui.container():
            st.header(f"📸 Capturing: {username}")
            st.markdown("Please look directly at the camera.")
            
            # Center the camera
            col_left, col_cam, col_right = st.columns([1, 4, 1])
            with col_cam:
                status_placeholder = st.empty()
                progress_bar       = st.progress(0, text="Starting camera…")
                frame_placeholder  = st.empty()

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
                    display_rgb,
                    channels="RGB",
                    output_format="JPEG",
                    width="stretch"
                )

            progress_bar.progress(
                min(count / CAPTURE_COUNT, 1.0),
                text=f"Captured {count} / {CAPTURE_COUNT}"
            )
            status_placeholder.info(status)

        # Run capture block
        result = capture_images(
            username,
            frame_callback=on_frame,
            stop_event=stop_event,
        )

        st.session_state.capture_result  = result
        st.session_state.capture_running = False
        
        # When capture is done, reload the page to restore the main setup UI
        st.rerun()

    # ── PERSISTENT RESULT ──────────────────────────────────────────────────
    # Rendered at the bottom of the active UI
    if not st.session_state.capture_running and st.session_state.get("capture_result"):
        res = st.session_state.capture_result
        if res["success"]:
            st.success(res["message"])
            st.info("✅ Capture complete. You can now **Train the Model** (next step).")
        else:
            st.error(res["message"])
