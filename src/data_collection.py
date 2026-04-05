import cv2
import os
import time
import threading
import numpy as np
from config import (
    DATA_PATH,
    CAPTURE_COUNT,
    CAPTURE_DELAY,
    BLUR_THRESHOLD,
    BRIGHTNESS_THRESHOLD,
)


def capture_images(
    username: str,
    frame_callback=None,
    stop_event: threading.Event = None,
) -> dict:

    clean_username = username.strip().lower().replace(" ", "_")
    save_path = os.path.join(DATA_PATH, clean_username)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {
            "success": False,
            "saved": 0,
            "path": save_path,
            "message": "Error: Could not access camera.",
        }

    # Headless-safe cascade path detection
    cascade_data = getattr(cv2, 'data', None)
    if cascade_data and hasattr(cascade_data, 'haarcascades'):
        cascade_path = cascade_data.haarcascades
    else:
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data')

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
    )
    if face_cascade.empty():
        # Final fallback
        face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

    count = 0
    last_capture_time = 0

    try:
        while count < CAPTURE_COUNT:
            # Respect external stop signal (e.g. Streamlit button press)
            if stop_event is not None and stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            is_valid = len(faces) == 1
            status_msg = "No Face Detected"
            color = (0, 0, 255)

            if is_valid:
                (x, y, w, h) = faces[0]
                face_roi = gray[y : y + h, x : x + w]

                blur_value = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                avg_brightness = np.mean(face_roi)

                if blur_value < BLUR_THRESHOLD:
                    status_msg = "Too Blurry – Hold Still"
                elif avg_brightness < BRIGHTNESS_THRESHOLD:
                    status_msg = "Too Dark – Increase Light"
                else:
                    status_msg = f"Capturing… ({count + 1}/{CAPTURE_COUNT})"
                    color = (0, 255, 0)

                    current_time = time.time()
                    if current_time - last_capture_time > CAPTURE_DELAY:
                        img_path = os.path.join(save_path, f"{count}.jpg")
                        cv2.imwrite(img_path, frame)
                        count += 1
                        last_capture_time = current_time

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Overlay progress text onto the frame for the callback preview
            cv2.putText(
                frame,
                f"Progress: {count}/{CAPTURE_COUNT}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                status_msg,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # Hand the annotated frame to the UI layer
            if frame_callback is not None:
                frame_callback(frame, status_msg, count)

    finally:
        cap.release()

    success = count == CAPTURE_COUNT
    return {
        "success": success,
        "saved": count,
        "path": save_path,
        "message": (
            f"Capture complete. {count} images saved to '{save_path}'."
            if success
            else f"Capture incomplete. Only {count}/{CAPTURE_COUNT} images saved."
        ),
    }


#  Standalone CLI entrypoint
if __name__ == "__main__":
    import sys

    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty.")
        sys.exit(1)

    def _preview(frame, status, count):
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    try:
        result = capture_images(username, frame_callback=_preview)
    except KeyboardInterrupt:
        result = {"success": False, "message": "Cancelled by user."}
    finally:
        cv2.destroyAllWindows()

    print(result["message"])
