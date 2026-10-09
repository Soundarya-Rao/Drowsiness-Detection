"""
Real-Time Facial Landmark & Region Extraction Test
Drowsiness Detection System for Indian Truck Drivers

Demonstrates live face mesh tracking, bounding box extraction for
both eyes and mouth, and real-time FPS performance monitoring.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.landmarks import (
    FaceLandmarkDetector,
    extract_eye_crops,
    extract_mouth_crop,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_INDICES
)


def draw_styled_bbox(img, bbox, label, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with rounded corner accents and text label."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, label_y - th - 3), (x1 + tw + 6, label_y + 3), color, -1)
    cv2.putText(img, label, (x1 + 3, label_y - 1), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)


def overlay_crop_pip(frame, crop, x_offset, y_offset, label, border_color=(255, 255, 255)):
    """Overlay a small crop preview in picture-in-picture style."""
    if crop is None or crop.size == 0:
        return
    ch, cw = crop.shape[:2]
    fh, fw = frame.shape[:2]

    if y_offset + ch > fh or x_offset + cw > fw:
        return

    # Draw border and insert crop
    cv2.rectangle(frame, (x_offset - 2, y_offset - 2), (x_offset + cw + 1, y_offset + ch + 1), border_color, 2)
    frame[y_offset:y_offset + ch, x_offset:x_offset + cw] = crop

    # Caption
    cv2.putText(
        frame,
        label,
        (x_offset, y_offset - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        border_color,
        1,
        cv2.LINE_AA
    )


def main():
    print("[*] Initializing MediaPipe FaceLandmarker...", flush=True)
    try:
        detector = FaceLandmarkDetector()
    except Exception as e:
        print(f"[!] ERROR initializing MediaPipe FaceLandmarker: {e}", flush=True)
        sys.exit(1)

    print("[*] Opening webcam...", flush=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] ERROR: Could not access webcam. Ensure no other application is using it.", flush=True)
        sys.exit(1)

    print("Setup OK", flush=True)
    print("[*] Webcam opened. Real-time landmark tracking active.", flush=True)
    print("[*] Click video window and press 'q' to quit.", flush=True)

    window_name = "Landmark Detection - Drowsiness Detection (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Performance tracking
    prev_time = time.perf_counter()
    fps_smooth = 0.0
    frame_count = 0
    fps_log_interval = 60  # Print to console every 60 frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame from webcam.", flush=True)
                break

            current_time = time.perf_counter()
            delta_time = current_time - prev_time
            prev_time = current_time

            # Compute smoothed FPS
            if delta_time > 0:
                current_fps = 1.0 / delta_time
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps if fps_smooth > 0 else current_fps

            frame_count += 1
            if frame_count % fps_log_interval == 0:
                ms_per_frame = (1000.0 / fps_smooth) if fps_smooth > 0 else 0
                status_text = "Real-time OK" if fps_smooth >= 20 else "Slow frame rate"
                print(
                    f"[*] [FPS Monitor] {fps_smooth:5.1f} FPS ({ms_per_frame:5.1f} ms/frame) - {status_text}",
                    flush=True
                )

            # Process frame with MediaPipe Face Mesh
            face_landmarks_list = detector.process(frame)

            h, w = frame.shape[:2]

            # Header HUD Background bar
            cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.line(frame, (0, 55), (w, 55), (60, 60, 60), 1)

            if face_landmarks_list and len(face_landmarks_list) > 0:
                landmarks = face_landmarks_list[0]

                # Status text (Face detected)
                cv2.putText(
                    frame,
                    "STATUS: Face Detected",
                    (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Extract eye crops and bounding boxes
                left_crop, right_crop, left_bbox, right_bbox = extract_eye_crops(
                    frame, landmarks, target_size=(96, 96)
                )

                # Extract mouth crop and bounding box
                mouth_crop, mouth_bbox = extract_mouth_crop(
                    frame, landmarks, target_size=(96, 96)
                )

                # Draw landmark points for eye and mouth contours
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    pt_x = int(round(landmarks[idx].x * w))
                    pt_y = int(round(landmarks[idx].y * h))
                    cv2.circle(frame, (pt_x, pt_y), 1, (0, 255, 255), -1)

                for idx in MOUTH_INDICES:
                    pt_x = int(round(landmarks[idx].x * w))
                    pt_y = int(round(landmarks[idx].y * h))
                    cv2.circle(frame, (pt_x, pt_y), 1, (255, 200, 0), -1)

                # Draw highlighted bounding boxes on face
                draw_styled_bbox(frame, left_bbox, "Left Eye", color=(0, 230, 255), thickness=2)
                draw_styled_bbox(frame, right_bbox, "Right Eye", color=(0, 230, 255), thickness=2)
                draw_styled_bbox(frame, mouth_bbox, "Mouth", color=(0, 165, 255), thickness=2)

                # Overlay picture-in-picture preview cards in top-right
                pip_y = 65
                overlay_crop_pip(frame, left_crop, w - 105, pip_y, "Left Eye (96x96)", border_color=(0, 230, 255))
                overlay_crop_pip(frame, right_crop, w - 105, pip_y + 115, "Right Eye (96x96)", border_color=(0, 230, 255))
                overlay_crop_pip(frame, mouth_crop, w - 105, pip_y + 230, "Mouth (96x96)", border_color=(0, 165, 255))

            else:
                # No face detected
                cv2.putText(
                    frame,
                    "STATUS: No Face Detected",
                    (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # FPS Display in top header
            fps_color = (0, 255, 0) if fps_smooth >= 24 else ((0, 255, 255) if fps_smooth >= 15 else (0, 0, 255))
            fps_label = f"FPS: {fps_smooth:4.1f}"
            (tw, _), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame,
                fps_label,
                (w - tw - 20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                fps_color,
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            # Exit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[*] 'q' key pressed. Closing cleanly...", flush=True)
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[*] Webcam and MediaPipe resources released. Ready for Phase 4!", flush=True)


if __name__ == "__main__":
    main()
