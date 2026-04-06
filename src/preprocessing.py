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
        """
        Align the face based on eye positions and crop to IMAGE_SIZE.
        If eyes are not detected, returns the simple crop of the face_rect.
        """
        (x, y, w, h) = face_rect
        roi_gray = gray_image[y : y + h, x : x + w]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_EYE_MIN_NEIGHBORS,
        )

        if len(eyes) >= 2:
            # Sort eyes by X coordinate
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            # Relative centers in ROI
            eye1_center = (ex1 + ew1 // 2, ey1 + eh1 // 2)
            eye2_center = (ex2 + ew2 // 2, ey2 + eh2 // 2)

            # Absolute centers in gray_image
            p1 = (x + eye1_center[0], y + eye1_center[1])
            p2 = (x + eye2_center[0], y + eye2_center[1])

            # Calculate angle and distance
            dY = p2[1] - p1[1]
            dX = p2[0] - p1[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Distance between eyes
            dist = np.sqrt(dX**2 + dY**2)
            if dist < 1:
                # Eyes are literally on top of each other, can't reliably align
                crop = gray_image[y : y + h, x : x + w]
                return cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            desired_dist = IMAGE_SIZE[0] * 0.4  # 40% of width
            scale = desired_dist / dist

            # Center of rotation is the midpoint between eyes
            center = (float((p1[0] + p2[0]) / 2), float((p1[1] + p2[1]) / 2))

            # Angle must be float
            angle = float(angle)
            scale = float(scale)

            M = cv2.getRotationMatrix2D(center, angle, scale)

            # Adjust translation to move eyes to the desired location
            # Desired eye midpoint should be at (width/2, height*0.35)
            tX = float(IMAGE_SIZE[0] * 0.5)
            tY = float(IMAGE_SIZE[1] * 0.35)
            M[0, 2] += (tX - center[0])
            M[1, 2] += (tY - center[1])

            # Warp the image to the target size
            face_aligned = cv2.warpAffine(
                gray_image,
                M,
                IMAGE_SIZE,
                flags=cv2.INTER_CUBIC
            )
            return face_aligned

        # Fallback: simple crop and resize if eyes not found
        crop = gray_image[y : y + h, x : x + w]
        return cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

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

        # 5. Alignment and Crop (Combined now)
        face_processed = self.align_face(gray, face_rect)

        # 6. Pixel normalisation [0, 1]
        return face_processed.astype("float32") / 255.0


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
