import cv2
import numpy as np
import os
from config import (
    IMAGE_SIZE,
    HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_EYE_MIN_NEIGHBORS,
)


class HaarPreprocessor:
    """Full face preprocessing pipeline using Haar cascades."""

    def __init__(self):
        # Headless-safe cascade path detection
        cascade_data = getattr(cv2, 'data', None)
        if cascade_data and hasattr(cascade_data, 'haarcascades'):
            cascade_path = cascade_data.haarcascades
        else:
            # Fallback for some headless distributions
            cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data')

        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_eye.xml")
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            # Final fallback: check common system paths
            fallbacks = ["/usr/share/opencv4/haarcascades", "/usr/local/share/opencv4/haarcascades"]
            for fb in fallbacks:
                self.face_cascade = cv2.CascadeClassifier(os.path.join(fb, "haarcascade_frontalface_default.xml"))
                self.eye_cascade = cv2.CascadeClassifier(os.path.join(fb, "haarcascade_eye.xml"))
                if not self.face_cascade.empty() and not self.eye_cascade.empty():
                    break
            
            if self.face_cascade.empty():
                raise IOError(f"Could not load Haar Cascade XML files. Tried: {cascade_path}")

    def normalize_lighting(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to improve contrast under varying lighting conditions."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def align_face(self, gray_image: np.ndarray, face_rect: tuple) -> np.ndarray:
        """Rotate image so the inter-eye axis is horizontal."""
        (x, y, w, h) = face_rect
        roi_gray = gray_image[y : y + h, x : x + w]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_EYE_MIN_NEIGHBORS,
        )

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            eye1_center = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
            eye2_center = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

            dY = eye2_center[1] - eye1_center[1]
            dX = eye2_center[0] - eye1_center[0]

            if dX == 0 and dY == 0:
                return gray_image

            angle = np.degrees(np.arctan2(dY, dX))
            midpoint = (
                (eye1_center[0] + eye2_center[0]) / 2,
                (eye1_center[1] + eye2_center[1]) / 2,
            )

            M = cv2.getRotationMatrix2D(midpoint, angle, 1.0)
            aligned = cv2.warpAffine(
                gray_image,
                M,
                (gray_image.shape[1], gray_image.shape[0]),
                flags=cv2.INTER_CUBIC,
            )
            return aligned

        return gray_image

    def process_image(self, image: np.ndarray):
        if image is None:
            return None

        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Lighting normalisation
        gray = self.normalize_lighting(gray)

        # 3. Face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )

        if len(faces) == 0:
            return None

        # 4. Largest face wins
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = face_rect

        # 5. Alignment
        aligned = self.align_face(gray, face_rect)

        # 6. Crop
        crop = aligned[y : y + h, x : x + w]

        # 7. Resize
        try:
            face_resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        except Exception:
            return None

        # 8. Pixel normalisation [0, 1]
        return face_resized.astype("float32") / 255.0


_preprocessor: HaarPreprocessor = None


def get_preprocessor() -> HaarPreprocessor:
    """Return the shared HaarPreprocessor instance, creating it on first call."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = HaarPreprocessor()
    return _preprocessor


def preprocess_image(image: np.ndarray):

    return get_preprocessor().process_image(image)


if __name__ == "__main__":
    import sys

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    print("Running headless test – reading 30 frames and counting detections…")
    detected = 0
    for _ in range(30):
        ret, frame = cap.read()
        if ret and preprocess_image(frame) is not None:
            detected += 1

    cap.release()
    print(f"Faces detected in {detected}/30 frames.")
