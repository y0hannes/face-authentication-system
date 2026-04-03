import os
import cv2
import numpy as np
import streamlit as st

from config import DATA_PATH


def init_session_state():
    defaults = {
        "capture_running": False,
        "auth_running": False,
        "capture_result": None,
        "last_prediction": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(frame)


def list_registered_users(data_path: str = None) -> list:
    root = data_path or DATA_PATH
    if not os.path.exists(root):
        return []
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def user_image_count(username: str, data_path: str = None) -> int:
    root = data_path or DATA_PATH
    user_dir = os.path.join(root, username.strip().lower().replace(" ", "_"))
    if not os.path.exists(user_dir):
        return 0
    return sum(
        1 for f in os.listdir(user_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
