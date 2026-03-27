import cv2
import os
import time
import numpy as np

def capture_images(username):
    clean_username = username.strip().lower().replace(" ", "_")
    save_path = os.path.join("data", "raw", clean_username)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    count = 0
    last_capture_time = 0
    
    BLUR_THRESHOLD = 50.0
    BRIGHTNESS_THRESHOLD = 80

    while count < 15:
        ret, frame = cap.read()
        if not ret: break

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
                if current_time - last_capture_time > 1.5:
                    img_path = os.path.join(save_path, f"{count}.jpg")
                    cv2.imwrite(img_path, frame)
                    count += 1
                    last_capture_time = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, f"Progress: {count}/15", (10, 30), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, status_msg, (10, 60), 1, 1.2, color, 2)
        
        cv2.imshow("High-Fidelity Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    return count == 15

if __name__ == "__main__":
    test_name = input("Enter System Username: ")
    capture_images(test_name)