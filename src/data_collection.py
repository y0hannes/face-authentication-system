import cv2
import os
import time
import numpy as np
from config import (
    DATA_PATH,
    IMAGE_SIZE,
    CAPTURE_COUNT,
    CAPTURE_DELAY,
    BLUR_THRESHOLD,
    BRIGHTNESS_THRESHOLD
)

def capture_images(username):
    clean_username = username.strip().lower().replace(" ", "_")
    save_path = os.path.join(DATA_PATH, clean_username)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return False

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    count = 0
    last_capture_time = 0

    while count < CAPTURE_COUNT:
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
            face_roi = gray[y:y+h, x:x+w]

            blur_value = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            avg_brightness = np.mean(face_roi)

            if blur_value < BLUR_THRESHOLD:
                status_msg = "Too Blurry - Hold Still"
            elif avg_brightness < BRIGHTNESS_THRESHOLD:
                status_msg = "Too Dark - Increase Light"
            else:
                status_msg = "Quality OK - Capturing..."
                color = (0, 255, 0)

                current_time = time.time()
                if current_time - last_capture_time > CAPTURE_DELAY:
                    # Resize image before saving
                    resized_frame = cv2.resize(frame, IMAGE_SIZE)

                    img_path = os.path.join(save_path, f"{count}.jpg")
                    cv2.imwrite(img_path, resized_frame)

                    count += 1
                    last_capture_time = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, f"Progress: {count}/{CAPTURE_COUNT}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, status_msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("High-Fidelity Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return count == CAPTURE_COUNT


if __name__ == "__main__":
    username = input("Enter System Username: ")
    success = capture_images(username)

    if success:
        print("Image capture completed successfully.")
    else:
        print("Image capture incomplete.")