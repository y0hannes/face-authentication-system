import os
import json
import datetime
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
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def user_image_count(username: str, data_path: str = None) -> int:
    root = data_path or DATA_PATH
    user_dir = os.path.join(root, username.strip().lower().replace(" ", "_"))
    if not os.path.exists(user_dir):
        return 0
    return sum(
        1 for f in os.listdir(user_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def delete_user(username: str, data_path: str = None) -> bool:
    import shutil

    root = data_path or DATA_PATH
    user_dir = os.path.join(root, username.strip().lower().replace(" ", "_"))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        return True
    return False

def get_login_attempts(data_path: str = None) -> int:
    root = data_path or DATA_PATH
    path = os.path.join(root, "login_attempts.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return int(f.read().strip())
            except Exception:
                return 0
    return 0

def increment_login_attempts(data_path: str = None):
    root = data_path or DATA_PATH
    path = os.path.join(root, "login_attempts.txt")
    current = get_login_attempts(root)
    with open(path, "w") as f:
        f.write(str(current + 1))


# ── Structured Authentication Event Log ───────────────────────────────────────

def log_auth_event(
    status: str,
    username: str | None,
    confidence: float | None,
    data_path: str = None,
) -> None:
    """Append one authentication event as a JSON line to the event log."""
    root = data_path or DATA_PATH
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "auth_events.jsonl")
    event = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status,          # "authenticated" | "unknown" | "no_face"
        "username": username or "Unknown",
        "confidence": round(confidence, 1) if confidence is not None else None,
    }
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def get_auth_events(data_path: str = None, limit: int = 50) -> list[dict]:
    """Return the most recent `limit` authentication events, newest first."""
    root = data_path or DATA_PATH
    path = os.path.join(root, "auth_events.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(events[-limit:]))


def get_failed_attempts(data_path: str = None) -> int:
    """Count events where status is not 'authenticated'."""
    events = get_auth_events(data_path=data_path, limit=10_000)
    return sum(1 for e in events if e.get("status") != "authenticated")


def get_weekly_login_counts(data_path: str = None) -> list[int]:
    """Return successful login counts for each of the last 4 weeks (oldest first)."""
    events = get_auth_events(data_path=data_path, limit=10_000)
    now = datetime.datetime.now()
    weeks = [0, 0, 0, 0]
    for e in events:
        if e.get("status") != "authenticated":
            continue
        try:
            ts = datetime.datetime.fromisoformat(e["ts"])
            delta_days = (now - ts).days
            week_idx = delta_days // 7  # 0 = current week, 1 = last week …
            if 0 <= week_idx < 4:
                weeks[3 - week_idx] += 1  # index 3 = most recent
        except Exception:
            continue
    return weeks


def get_outcome_counts(data_path: str = None) -> dict:
    """Return totals for each outcome: authenticated, unknown (face not recognised)."""
    events = get_auth_events(data_path=data_path, limit=10_000)
    counts = {"authenticated": 0, "unknown": 0}
    for e in events:
        status = e.get("status", "unknown")
        if status == "authenticated":
            counts["authenticated"] += 1
        else:
            counts["unknown"] += 1
    return counts


def get_per_user_login_counts(data_path: str = None) -> dict:
    """Return a dict of {username: login_count} for authenticated events only."""
    events = get_auth_events(data_path=data_path, limit=10_000)
    counts: dict[str, int] = {}
    for e in events:
        if e.get("status") == "authenticated":
            user = e.get("username", "Unknown")
            counts[user] = counts.get(user, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
