import cv2
import numpy as np
import os
from config import IMAGE_SIZE

class HaarPreprocessor:
    def __init__(self):
        cascade_path = cv2.data.haarcascades

        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml')
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise IOError("Could not load Haar Cascade XML files.")

    def normalize_lighting(self, gray_image):
        """Improve lighting using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def align_face(self, gray_image, face_rect):
        """Align face using detected eyes"""
        (x, y, w, h) = face_rect
        roi_gray = gray_image[y:y+h, x:x+w]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10
        )

        if len(eyes) >= 2:
            # Take two eyes with smallest x (leftmost)
            eyes = sorted(eyes, key=lambda e: e[0])[:2]

            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            eye1_center = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
            eye2_center = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

            dY = eye2_center[1] - eye1_center[1]
            dX = eye2_center[0] - eye1_center[0]

            # Avoid division by zero
            if dX == 0 and dY == 0:
                return gray_image

            angle = np.degrees(np.arctan2(dY, dX))
            midpoint = ((eye1_center[0] + eye2_center[0]) / 2,
                        (eye1_center[1] + eye2_center[1]) / 2)

            # Make sure midpoint is tuple of floats
            if not (isinstance(midpoint, tuple) and len(midpoint) == 2):
                return gray_image

            M = cv2.getRotationMatrix2D(midpoint, angle, 1.0)
            aligned = cv2.warpAffine(
                gray_image,
                M,
                (gray_image.shape[1], gray_image.shape[0]),
                flags=cv2.INTER_CUBIC
            )

            return aligned

        return gray_image

    def process_image(self, image):
        """
        Full pipeline:
        Gray → Lighting → Detect → Align → Crop → Resize → Normalize
        """
        if image is None:
            return None

        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Normalize lighting
        gray = self.normalize_lighting(gray)

        # 3. Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return None

        # 4. Take largest face
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = face_rect

        # 5. Align face
        aligned = self.align_face(gray, face_rect)

        # 6. Crop
        crop = aligned[y:y+h, x:x+w]

        # 7. Resize
        try:
            face_resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        except Exception:
            return None

        # 8. Normalize pixels (0–1)
        face_final = face_resized.astype("float32") / 255.0

        return face_final

# GLOBAL INSTANCE
_preprocessor = HaarPreprocessor()

def preprocess_image(image):
    """Interface for pipeline: returns (64,64) or None"""
    return _preprocessor.process_image(image)


# --- OPTIONAL TEST ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit...")

    while True:
        ret, frame = cap.read()
        processed = preprocess_image(frame)

        if processed is not None:
            display = (processed * 255).astype(np.uint8)
            cv2.imshow("Processed Face", display)

        cv2.imshow("Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()