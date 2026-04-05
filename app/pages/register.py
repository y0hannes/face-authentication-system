import time
import os
import cv2
import streamlit as st

from config import CAPTURE_COUNT, DATA_PATH
from src.train import train_and_save
from src.predict import reload_model
from app.utils import init_session_state


def _file_to_cv2(uploaded_file):
    import numpy as np
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)


def show():
    init_session_state()

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
        .cloud-info-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(0,96,171,0.08); padding: 6px 14px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: #0060ab;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='font-weight:900;letter-spacing:-0.05em;color:var(--txt-main);font-size:2.25rem;margin-bottom:0'>Create Your Secure Profile</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0060ab;font-weight:500;font-size:1.125rem;margin-bottom:1.5rem;margin-top:4px'>Cloud-ready biometric enrollment via browser cam.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 5], gap="large")

    if "reg_captured_samples" not in st.session_state:
        st.session_state.reg_captured_samples = 0

    with col_left:
        st.markdown("<div class='camera-label'>Browser Webcam</div>", unsafe_allow_html=True)
        
        captured_file = st.camera_input("Enrollment Photo")
        
        st.markdown(f"""
            <div class="cloud-info-badge">
                Cloud Mode · {st.session_state.reg_captured_samples} Samples Stored
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-box'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:var(--brand)'>Target</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>Min 1</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box' style='border-left-color:#00685a'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:#00685a'>Status</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>Active</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-box' style='border-left-color:#0060ab'><div style='font-size:10px;font-weight:700;text-transform:uppercase;color:#0060ab'>Type</div><div style='font-size:1.4rem;font-weight:800;color:var(--txt-main)'>Cloud</div></div>", unsafe_allow_html=True)

    with col_right:
        username = st.text_input("FULL NAME", placeholder="e.g. Dr. Julian Thorne", key="reg_name_input")
        
        st.info("🔒 **Browser Security**\n\nThe web browser will request permission to access your camera. Photos are processed directly in your secure session.")

        # Handle 'Save Frame' logic
        if captured_file and username:
            if st.button("📥 Save This Sample", width="stretch", type="primary"):
                clean_username = username.strip().lower().replace(" ", "_")
                save_path = os.path.join(DATA_PATH, clean_username)
                os.makedirs(save_path, exist_ok=True)
                
                # Find next index
                existing = [f for f in os.listdir(save_path) if f.lower().endswith(".jpg")]
                idx = len(existing)
                
                frame = _file_to_cv2(captured_file)
                img_path = os.path.join(save_path, f"{idx}.jpg")
                cv2.imwrite(img_path, frame)
                
                st.session_state.reg_captured_samples = idx + 1
                st.success(f"Sample {idx + 1} saved!")
                time.sleep(1)
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.session_state.reg_captured_samples > 0:
            if st.button("🚀 Synchronize Neural Matrix", width="stretch"):
                with st.spinner("Compiling biometric model..."):
                    train_res = train_and_save()
                if train_res.get("success"):
                    reload_model()
                    st.success(train_res["message"])
                    st.session_state.reg_captured_samples = 0 # Reset for next
                else:
                    st.error(train_res.get("message", "Training failed."))

        st.markdown("<p style='text-align:center;font-size:12px;color:var(--txt-sub);margin-top:16px'>Click 'Save This Sample' multiple times for better accuracy.</p>", unsafe_allow_html=True)
        st.warning("💡 **Tip:** Take 1-3 photos while tilting your head slightly to improve recognition.")
