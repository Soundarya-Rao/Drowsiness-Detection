"""
Facial Landmark Detection and Region Extraction Module
Drowsiness Detection System for Indian Truck Drivers

Uses MediaPipe FaceLandmarker (Tasks API) to locate facial landmarks
and extract cropped eye and mouth regions for downstream classification.
"""

import os
import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canonical MediaPipe Face Mesh landmark indices
# Left Eye (anatomical left / camera right)
LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]

# Right Eye (anatomical right / camera left)
RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

# Mouth & Lips (outer boundary + inner opening)
MOUTH_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]

# 6-point Eye Aspect Ratio (EAR) Landmark Indices: [p1, p2, p3, p4, p5, p6]
# Left Eye (anatomical left / camera right)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
# Right Eye (anatomical right / camera left)
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]


def compute_ear(landmarks, indices: list[int], img_w: int, img_h: int) -> float:
    """
    Compute Eye Aspect Ratio (EAR) using standard 6-point formula:
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h], dtype=np.float32) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts

    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h_dist = float(np.linalg.norm(p1 - p4))

    if h_dist < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h_dist))


# Inner-lip Mouth Aspect Ratio (MAR) Landmark Indices:
# Vertical pairs: (82, 87), (13, 14), (312, 317)
# Horizontal pair: (78, 308)
MOUTH_MAR_VERTICAL_PAIRS = [(82, 87), (13, 14), (312, 317)]
MOUTH_MAR_HORIZONTAL_PAIR = (78, 308)


def compute_mar(landmarks, img_w: int, img_h: int) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) using MediaPipe inner-lip landmarks:
    Vertical pairs: (82, 87), (13, 14), (312, 317)
    Horizontal pair: (78, 308)
    Formula: mean(three vertical distances) / horizontal distance
    """
    p78 = np.array([landmarks[78].x * img_w, landmarks[78].y * img_h], dtype=np.float32)
    p308 = np.array([landmarks[308].x * img_w, landmarks[308].y * img_h], dtype=np.float32)
    h_dist = float(np.linalg.norm(p78 - p308))

    if h_dist < 1e-6:
        return 0.0

    v1 = float(np.linalg.norm(
        np.array([landmarks[82].x * img_w, landmarks[82].y * img_h], dtype=np.float32) -
        np.array([landmarks[87].x * img_w, landmarks[87].y * img_h], dtype=np.float32)
    ))
    v2 = float(np.linalg.norm(
        np.array([landmarks[13].x * img_w, landmarks[13].y * img_h], dtype=np.float32) -
        np.array([landmarks[14].x * img_w, landmarks[14].y * img_h], dtype=np.float32)
    ))
    v3 = float(np.linalg.norm(
        np.array([landmarks[312].x * img_w, landmarks[312].y * img_h], dtype=np.float32) -
        np.array([landmarks[317].x * img_w, landmarks[317].y * img_h], dtype=np.float32)
    ))

    mean_vertical = (v1 + v2 + v3) / 3.0
    return float(mean_vertical / h_dist)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def ensure_face_landmarker_model() -> Path:
    """Download the official MediaPipe FaceLandmarker model task bundle if missing."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Downloading MediaPipe face landmarker model to '{MODEL_PATH}'...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Model download complete.")
    return MODEL_PATH


class FaceLandmarkDetector:
    """
    Real-time face landmarker wrapper using MediaPipe Tasks API.
    Optimized for high-FPS live webcam feeds.
    """

    def __init__(self, model_path: Path | str | None = None, num_faces: int = 1):
        if model_path is None:
            model_path = ensure_face_landmarker_model()
        else:
            model_path = Path(model_path)

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray):
        """
        Process a BGR OpenCV frame and return landmark detection results.
        Returns list of landmark lists (one per face detected).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mp_image)
        return detection_result.face_landmarks

    def close(self):
        """Release underlying MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()


def get_bounding_box(
    landmarks,
    indices: list[int],
    img_w: int,
    img_h: int,
    margin_x: float = 0.25,
    margin_y: float = 0.35,
    make_square: bool = True
) -> tuple[int, int, int, int]:
    """
    Compute pixel bounding box around specified landmark indices with margins.
    Returns (x1, y1, x2, y2) clamped to image boundaries.
    """
    pts_x = [landmarks[i].x * img_w for i in indices]
    pts_y = [landmarks[i].y * img_h for i in indices]

    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)

    w = max_x - min_x
    h = max_y - min_y

    # Add margins
    pad_w = w * margin_x
    pad_h = h * margin_y

    x1 = min_x - pad_w
    x2 = max_x + pad_w
    y1 = min_y - pad_h
    y2 = max_y + pad_h

    # Optionally make bounding box square for neural net inputs
    if make_square:
        cur_w = x2 - x1
        cur_h = y2 - y1
        max_dim = max(cur_w, cur_h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        x1 = cx - max_dim / 2.0
        x2 = cx + max_dim / 2.0
        y1 = cy - max_dim / 2.0
        y2 = cy + max_dim / 2.0

    # Clamp to frame dimensions
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


def crop_region(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: tuple[int, int] | None = None
) -> np.ndarray | None:
    """
    Extract pixel crop from frame using bounding box (x1, y1, x2, y2).
    Optionally resizes to target_size (e.g. (96, 96)).
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if target_size is not None and (crop.shape[1] != target_size[0] or crop.shape[0] != target_size[1]):
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    return crop


def extract_eye_crops(
    frame: np.ndarray,
    landmarks,
    target_size: tuple[int, int] = (96, 96)
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Extract Left and Right eye patches from frame.
    Returns: (left_crop, right_crop, left_bbox, right_bbox)
    """
    h, w = frame.shape[:2]

    # Calculate bounding boxes
    left_bbox = get_bounding_box(landmarks, LEFT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)
    right_bbox = get_bounding_box(landmarks, RIGHT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)

    left_crop = crop_region(frame, left_bbox, target_size=target_size)
    right_crop = crop_region(frame, right_bbox, target_size=target_size)

    return left_crop, right_crop, left_bbox, right_bbox


def extract_mouth_crop(
    frame: np.ndarray,
    landmarks,
    target_size: tuple[int, int] = (96, 96)
) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """
    Extract mouth / yawn patch from frame.
    Returns: (mouth_crop, mouth_bbox)
    """
    h, w = frame.shape[:2]
    mouth_bbox = get_bounding_box(landmarks, MOUTH_INDICES, w, h, margin_x=0.25, margin_y=0.30, make_square=True)
    mouth_crop = crop_region(frame, mouth_bbox, target_size=target_size)
    return mouth_crop, mouth_bbox


# Module-level CLAHE instance for lighting robustness (created once to avoid per-frame overhead)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_eye_crop_for_model(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess an eye crop for eye_model.keras.
    Converts BGR to 3-channel grayscale (R=G=B) with CLAHE equalization
    to match the MRL infrared training distribution, improve lighting robustness,
    and eliminate webcam chroma/color channel mismatch.
    """
    if crop_bgr.ndim == 2:
        gray = crop_bgr
    elif crop_bgr.shape[2] == 4:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB).astype(np.float32)

