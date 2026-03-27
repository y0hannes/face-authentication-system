import cv2
import os

def capture_images(username):
    clean_username = username.strip().lower().replace(" ", "_")
    save_path = os.path.join("data", "raw", clean_username)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam access failed.")
        return False

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    count = 0
    print(f"\n--- Registration Started: {clean_username} ---")

    while count < 15:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        is_valid = len(faces) == 1
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        
        cv2.putText(frame, f"Captured: {count}/15", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if is_valid:
                img_path = os.path.join(save_path, f"{count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Saved: {img_path}")
                count += 1
            else:
                print("QC Failed")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 15:
        print(f"\n✅ Success! 15 images saved in '{save_path}'")
        return True
    return False

if __name__ == "__main__":
    test_name = input("Enter username: ")
    capture_images(test_name)