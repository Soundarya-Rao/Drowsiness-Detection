"""
Drowsiness & Fatigue Detection Engine
Reusable core detection engine decoupling video/audio processing from UI/CLI.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.landmarks import (
    FaceLandmarkDetector,
    get_bounding_box,
    crop_region,
    preprocess_eye_crop_for_model,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_INDICES,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    compute_ear,
    compute_mar
)
from src.audio_alerts import AudioManager


def get_timestamp_str() -> str:
    """Return formatted timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_str() -> str:
    """Return compact time string (HH:MM:SS) for UI logs."""
    return datetime.now().strftime("%H:%M:%S")


def draw_styled_bbox(img, bbox, label, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with label tag."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, label_y - th - 3), (x1 + tw + 6, label_y + 3), color, -1)
    cv2.putText(img, label, (x1 + 3, label_y - 1), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)


def overlay_crop_pip(frame, crop, x_offset, y_offset, label, border_color=(255, 255, 255)):
    """Render a cropped image patch in picture-in-picture style."""
    if crop is None or crop.size == 0:
        return
    ch, cw = crop.shape[:2]
    fh, fw = frame.shape[:2]

    if y_offset + ch > fh or x_offset + cw > fw:
        return

    cv2.rectangle(frame, (x_offset - 2, y_offset - 2), (x_offset + cw + 1, y_offset + ch + 1), border_color, 2)
    frame[y_offset:y_offset + ch, x_offset:x_offset + cw] = crop

    cv2.putText(
        frame,
        label,
        (x_offset, y_offset - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        border_color,
        1,
        cv2.LINE_AA
    )


def draw_drowsy_banner(frame, closed_duration: float, ear_val: float | None = None):
    """Draw a prominent, high-visibility red alert banner across the screen."""
    h, w = frame.shape[:2]
    banner_h = 100
    banner_y1 = h // 2 - banner_h // 2
    banner_y2 = banner_y1 + banner_h

    # Semi-transparent red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Gold/Yellow border
    cv2.rectangle(frame, (20, banner_y1), (w - 20, banner_y2), (0, 215, 255), 3)

    # Alert Title Text
    title_text = "*** DROWSY ALERT! WAKE UP! ***"
    (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    title_x = (w - tw1) // 2
    cv2.putText(
        frame,
        title_text,
        (title_x, banner_y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    # Duration Subtitle Text
    if ear_val is not None:
        sub_text = f"Eyes continuously closed for {closed_duration:.1f}s (EAR: {ear_val:.3f})"
    else:
        sub_text = f"Eyes continuously closed for {closed_duration:.1f}s"
    (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    sub_x = (w - tw2) // 2
    cv2.putText(
        frame,
        sub_text,
        (sub_x, banner_y1 + 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_fatigue_warning_banner(frame):
    """Draw a prominent amber/orange fatigue warning banner across the screen."""
    h, w = frame.shape[:2]
    banner_h = 90
    banner_y1 = h // 2 - banner_h // 2
    banner_y2 = banner_y1 + banner_h

    # Semi-transparent amber overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (0, 100, 220), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Amber/yellow border
    cv2.rectangle(frame, (20, banner_y1), (w - 20, banner_y2), (0, 215, 255), 3)

    # Warning Title Text
    title_text = "*** FATIGUE WARNING: TAKE A BREAK ***"
    (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    title_x = (w - tw1) // 2
    cv2.putText(
        frame,
        title_text,
        (title_x, banner_y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Subtitle
    sub_text = "Frequent yawning detected. Pull over safely and rest."
    (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    sub_x = (w - tw2) // 2
    cv2.putText(
        frame,
        sub_text,
        (sub_x, banner_y1 + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


def draw_no_face_alarm_banner(frame, absent_duration: float):
    """Draw a prominent red alert banner when driver is not visible and siren escalates."""
    h, w = frame.shape[:2]
    banner_h = 100
    banner_y1 = h // 2 - banner_h // 2
    banner_y2 = banner_y1 + banner_h

    # Semi-transparent red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Gold/Yellow border
    cv2.rectangle(frame, (20, banner_y1), (w - 20, banner_y2), (0, 215, 255), 3)

    # Alert Title Text
    title_text = "*** DRIVER NOT VISIBLE! SIREN ACTIVE! ***"
    (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.88, 3)
    title_x = (w - tw1) // 2
    cv2.putText(
        frame,
        title_text,
        (title_x, banner_y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    # Duration Subtitle Text
    sub_text = f"No face detected for {absent_duration:.1f}s. Return eyes to road immediately."
    (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    sub_x = (w - tw2) // 2
    cv2.putText(
        frame,
        sub_text,
        (sub_x, banner_y1 + 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_calibration_card(
    frame,
    elapsed: float,
    duration: float,
    face_detected: bool,
    frame_ear: float | None = None,
    frame_mar: float | None = None
):
    """Draw a centered calibration card instructing driver to look with eyes open and mouth closed."""
    h, w = frame.shape[:2]
    card_w, card_h = 580, 130
    cx, cy = w // 2, h // 2
    x1 = cx - card_w // 2
    y1 = cy - card_h // 2
    x2 = x1 + card_w
    y2 = y1 + card_h

    # Semi-transparent dark overlay card
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    border_color = (0, 215, 255) if face_detected else (0, 140, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    # Primary Instruction Title
    title = "LOOK AT CAMERA: EYES OPEN, MOUTH CLOSED"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, title, (cx - tw // 2, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Subtitle with countdown and current live EAR & MAR
    remaining = max(0.0, duration - elapsed)
    if face_detected and frame_ear is not None and frame_mar is not None:
        sub = f"Calibrating baselines... {remaining:.1f}s left (Live EAR: {frame_ear:.3f} | MAR: {frame_mar:.3f})"
    elif face_detected and frame_ear is not None:
        sub = f"Calibrating baselines... {remaining:.1f}s left (Live EAR: {frame_ear:.3f})"
    elif face_detected:
        sub = f"Calibrating baselines... {remaining:.1f}s left"
    else:
        sub = "Please center your face in view of the camera"

    (stw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, sub, (cx - stw // 2, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 1, cv2.LINE_AA)

    # Progress Bar
    pb_w = card_w - 60
    pb_x1 = x1 + 30
    pb_y1 = y1 + 85
    pb_h = 16
    cv2.rectangle(frame, (pb_x1, pb_y1), (pb_x1 + pb_w, pb_y1 + pb_h), (70, 70, 70), 1)

    pct = min(1.0, max(0.0, elapsed / duration)) if duration > 0 else 1.0
    fill_w = int(pb_w * pct)
    if fill_w > 0:
        cv2.rectangle(frame, (pb_x1 + 1, pb_y1 + 1), (pb_x1 + fill_w, pb_y1 + pb_h - 1), (0, 215, 255), -1)


@dataclass
class EngineState:
    """Snapshot of detection metrics, status, and alerts for a single frame."""
    status: str                         # "AWAKE", "ONE_EYE_CLOSED", "BOTH_EYES_CLOSED", "DROWSY", "YAWNING", "CALIBRATING", "NO_FACE"
    status_text: str                    # Human-readable label
    status_color: tuple[int, int, int]  # (B, G, R) color tuple
    face_detected: bool = False
    no_face_duration: float = 0.0
    no_face_alarm_active: bool = False
    is_calibrating: bool = True
    calibration_progress: float = 0.0   # 0.0 to 1.0
    calibration_remaining: float = 0.0  # seconds
    baseline_ear: float | None = None
    baseline_mar: float | None = None
    effective_ear_threshold: float = 0.20
    frame_ear: float | None = None
    left_ear: float | None = None
    right_ear: float | None = None
    frame_mar: float | None = None
    yawn_count: int = 0
    is_yawning: bool = False
    yawn_duration: float = 0.0
    drowsy_flag: bool = False
    closed_duration: float = 0.0
    consecutive_closed_frames: int = 0
    fatigue_warning_flag: bool = False
    fps: float = 0.0
    sound_status: str = "SOUND: ON"     # "SOUND: ON", "SOUND: MUTED", "SOUND: OFF"
    is_muted: bool = False
    volume: float = 1.0
    events: list[dict] = field(default_factory=list)


class DrowsinessEngine:
    """
    Core stateful processing engine for Eye & Yawn Drowsiness Detection.
    Decoupled from presentation, reusable across CLI and GUI.
    """

    def __init__(
        self,
        threshold_seconds: float = 1.5,
        calibration_seconds: float = 3.0,
        ear_ratio: float = 0.68,
        mar_threshold: float = 0.60,
        yawn_seconds: float = 1.2,
        yawn_warning_count: int = 3,
        yawn_warning_window: float = 600.0,
        no_sound: bool = False,
        volume: float = 1.0,
        show_cnn: bool = False,
        eye_threshold: float = 0.50,
        model_path: str = "models/eye_model.keras",
        mirror: bool = True,
        draw_overlays: bool = True,
        log_to_console: bool = True,
        no_face_seconds: float = 3.0,
        no_face_alarm_seconds: float = 8.0
    ):
        self.threshold_seconds = float(threshold_seconds)
        self.calibration_seconds = float(calibration_seconds)
        self.ear_ratio = float(ear_ratio)
        self.mar_threshold = float(mar_threshold)
        self.yawn_seconds = float(yawn_seconds)
        self.yawn_warning_count = int(yawn_warning_count)
        self.yawn_warning_window = float(yawn_warning_window)
        self.no_sound = bool(no_sound)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.show_cnn = bool(show_cnn)
        self.eye_threshold = float(eye_threshold)
        self.model_path = Path(model_path)
        self.mirror = bool(mirror)
        self.draw_overlays = bool(draw_overlays)
        self.log_to_console = bool(log_to_console)
        self.no_face_seconds = float(no_face_seconds)
        self.no_face_alarm_seconds = float(no_face_alarm_seconds)

        # Driver Visibility Safeguard State
        self.no_face_start_time: float | None = None
        self.no_face_active: bool = False
        self.no_face_alarm_active: bool = False
        self.last_no_face_beep_time: float | None = None
        self.no_face_duration: float = 0.0

        # 1. Initialize Face Landmark Detector
        self.detector = FaceLandmarkDetector()

        # 2. Initialize Audio Manager
        self.audio_manager = AudioManager(no_sound=self.no_sound, volume=self.volume)

        # 3. Optional CNN Model
        self.eye_model = None
        if self.show_cnn and self.model_path.exists():
            try:
                import tensorflow as tf
                self.eye_model = tf.keras.models.load_model(str(self.model_path))
                if self.log_to_console:
                    print(f"[+] Loaded eye model from '{self.model_path}'.", flush=True)
            except Exception as e:
                if self.log_to_console:
                    print(f"[!] Warning: Could not load eye model: {e}", flush=True)

        # Calibration State
        self.is_calibrating = True
        self.calibration_start_time = None
        self.calibration_ears: list[float] = []
        self.calibration_mars: list[float] = []
        self.baseline_ear: float | None = None
        self.baseline_mar: float | None = None
        self.ear_threshold: float | None = None
        self.fallback_ear_threshold: float = 0.20

        # Yawn State
        self.yawn_count = 0
        self.yawn_start_time = None
        self.yawn_duration = 0.0
        self.is_yawning = False

        # Fatigue Warning State
        self.yawn_timestamps: list[float] = []
        self.fatigue_warning_active = False
        self.fatigue_warning_start_time: float | None = None
        self.fatigue_display_duration = 5.0

        # Eye Closure & Alert State
        self.consecutive_closed_frames = 0
        self.closed_start_time = None
        self.closed_duration = 0.0
        self.drowsy_flag = False
        self.prev_drowsy_flag = False

        # FPS Tracking
        self.prev_frame_time = time.perf_counter()
        self.fps_smooth = 15.0

    def recalibrate(self):
        """Reset calibration baselines and trigger a new calibration cycle."""
        self.is_calibrating = True
        self.calibration_start_time = None
        self.calibration_ears.clear()
        self.calibration_mars.clear()
        self.baseline_ear = None
        self.baseline_mar = None
        self.ear_threshold = None
        self.consecutive_closed_frames = 0
        self.closed_start_time = None
        self.closed_duration = 0.0
        self.drowsy_flag = False
        self.no_face_active = False
        self.no_face_alarm_active = False
        self.no_face_start_time = None
        self.last_no_face_beep_time = None
        self.no_face_duration = 0.0
        self.audio_manager.stop_alarm()
        if self.log_to_console:
            print("[*] Recalibration initiated: Look at camera with eyes open and mouth closed.", flush=True)

    def set_threshold_seconds(self, seconds: float):
        """Update eye-closure duration required to trigger drowsy alert."""
        self.threshold_seconds = max(0.5, float(seconds))

    def set_mar_threshold(self, thresh: float):
        """Update Mouth Aspect Ratio threshold for yawn detection."""
        self.mar_threshold = max(0.2, min(1.2, float(thresh)))

    def set_volume(self, volume: float):
        """Update audio alert volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, float(volume)))
        self.audio_manager.set_volume(self.volume)

    def toggle_mute(self) -> bool:
        """Toggle mute state of audio manager and return new muted state."""
        return self.audio_manager.toggle_mute()

    def set_mute(self, muted: bool):
        """Set mute state directly."""
        self.audio_manager.set_mute(muted)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, EngineState]:
        """
        Process a single video frame.
        Computes landmarks, EAR, MAR, closure timing, yawns, fatigue alerts, and audio.
        Returns:
            annotated_frame: Copy of frame with visual overlays (or clean mirrored if draw_overlays=False)
            state: EngineState snapshot
        """
        events = []

        # 1. Mirror View
        if self.mirror:
            frame = cv2.flip(frame, 1)
        else:
            frame = frame.copy()

        current_time = time.perf_counter()
        dt = current_time - self.prev_frame_time
        self.prev_frame_time = current_time

        # Update smoothed FPS
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_smooth = 0.90 * self.fps_smooth + 0.10 * current_fps

        dynamic_frame_threshold = max(3, int(round(self.fps_smooth * self.threshold_seconds)))

        # Update fatigue warning display timer
        if self.fatigue_warning_active:
            if self.fatigue_warning_start_time is not None and (current_time - self.fatigue_warning_start_time) >= self.fatigue_display_duration:
                self.fatigue_warning_active = False

        h, w = frame.shape[:2]

        # Landmark Detection
        landmarks_list = self.detector.process(frame)

        face_detected = False
        screen_left_ear: float | None = None
        screen_right_ear: float | None = None
        frame_ear: float | None = None
        frame_mar: float | None = None
        screen_left_closed = False
        screen_right_closed = False
        both_closed = False

        screen_left_bbox = None
        screen_right_bbox = None
        mouth_bbox = None
        screen_left_crop = None
        screen_right_crop = None

        cnn_left_score: float | None = None
        cnn_right_score: float | None = None

        calib_elapsed = 0.0
        calib_progress = 0.0
        calib_remaining = 0.0

        if landmarks_list and len(landmarks_list) > 0:
            face_detected = True
            landmarks = landmarks_list[0]

            # Compute EAR & Bounding Boxes
            ear_1 = compute_ear(landmarks, LEFT_EYE_EAR_INDICES, w, h)
            ear_2 = compute_ear(landmarks, RIGHT_EYE_EAR_INDICES, w, h)

            bbox_1 = get_bounding_box(landmarks, LEFT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)
            bbox_2 = get_bounding_box(landmarks, RIGHT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)

            # Screen-relative eye designation: box with smaller x center is Screen "L"
            cx_1 = (bbox_1[0] + bbox_1[2]) / 2.0
            cx_2 = (bbox_2[0] + bbox_2[2]) / 2.0

            if cx_1 < cx_2:
                screen_left_bbox = bbox_1
                screen_left_ear = ear_1
                screen_right_bbox = bbox_2
                screen_right_ear = ear_2
            else:
                screen_left_bbox = bbox_2
                screen_left_ear = ear_2
                screen_right_bbox = bbox_1
                screen_right_ear = ear_1

            frame_ear = (screen_left_ear + screen_right_ear) / 2.0

            # Compute MAR & Mouth Bounding Box
            frame_mar = compute_mar(landmarks, w, h)
            mouth_bbox = get_bounding_box(landmarks, MOUTH_INDICES, w, h, margin_x=0.25, margin_y=0.30, make_square=True)

            # Crop 96x96 eye patches
            screen_left_crop = crop_region(frame, screen_left_bbox, target_size=(96, 96))
            screen_right_crop = crop_region(frame, screen_right_bbox, target_size=(96, 96))

            # Optional CNN
            if self.show_cnn and self.eye_model is not None and screen_left_crop is not None and screen_right_crop is not None:
                try:
                    left_input = preprocess_eye_crop_for_model(screen_left_crop)
                    right_input = preprocess_eye_crop_for_model(screen_right_crop)
                    eye_batch = np.stack([left_input, right_input], axis=0)
                    preds = self.eye_model(eye_batch, training=False).numpy()
                    cnn_left_score = float(preds[0][0])
                    cnn_right_score = float(preds[1][0])
                except Exception:
                    pass

            # Calibration Phase
            if self.is_calibrating:
                if self.calibration_start_time is None:
                    self.calibration_start_time = current_time
                    events.append({
                        "type": "CALIBRATION_START",
                        "message": "Calibration started (eyes open, mouth closed)",
                        "time": get_time_str()
                    })

                calib_elapsed = current_time - self.calibration_start_time
                calib_progress = min(1.0, max(0.0, calib_elapsed / self.calibration_seconds))
                calib_remaining = max(0.0, self.calibration_seconds - calib_elapsed)

                self.calibration_ears.append(frame_ear)
                self.calibration_mars.append(frame_mar)

                if calib_elapsed >= self.calibration_seconds and len(self.calibration_ears) >= 10:
                    self.baseline_ear = float(np.median(self.calibration_ears))
                    self.baseline_mar = float(np.median(self.calibration_mars))
                    self.ear_threshold = float(self.baseline_ear * self.ear_ratio)
                    self.is_calibrating = False

                    timestamp = get_timestamp_str()
                    calib_msg = (
                        f"Calibration complete: Baseline EAR={self.baseline_ear:.3f}, "
                        f"Closed Thresh={self.ear_threshold:.3f}, Baseline MAR={self.baseline_mar:.3f}"
                    )
                    events.append({
                        "type": "CALIBRATION_DONE",
                        "message": calib_msg,
                        "time": get_time_str()
                    })

                    if self.log_to_console:
                        print(
                            f"[{timestamp}] [CALIBRATION] Baseline Open EAR: {self.baseline_ear:.3f} | "
                            f"Closed Eye Threshold ({int(self.ear_ratio * 100)}%): {self.ear_threshold:.3f} | "
                            f"Baseline Closed MAR: {self.baseline_mar:.3f} (Yawn Threshold: {self.mar_threshold:.2f})",
                            flush=True
                        )

        # -------------------------------------------------------------
        # Driver Visibility Safeguard Logic
        # -------------------------------------------------------------
        if face_detected:
            # Face is present: check if recovering from NO_FACE state
            if self.no_face_active:
                # Stop alarm audio immediately
                self.audio_manager.stop_alarm()
                timestamp = get_timestamp_str()
                stop_msg = f"Driver visible again (absent for {self.no_face_duration:.1f}s)"
                events.append({
                    "type": "NO_FACE_STOP",
                    "message": stop_msg,
                    "time": get_time_str()
                })
                if self.log_to_console:
                    print(
                        f"[{timestamp}] [INFO]  NO_FACE_STOP: Driver visible again after {self.no_face_duration:.1f}s",
                        flush=True
                    )
                self.no_face_active = False
                self.no_face_alarm_active = False

            # Reset absence timer (short loss < no_face_seconds silently resets here, keeping baselines)
            self.no_face_start_time = None
            self.last_no_face_beep_time = None
            self.no_face_duration = 0.0
        else:
            # Face NOT detected: track real-time absence duration
            if self.no_face_start_time is None:
                self.no_face_start_time = current_time
            self.no_face_duration = current_time - self.no_face_start_time

            # Short loss (< no_face_seconds, e.g. < 3.0s): do NOT trigger anything!
            if self.no_face_duration >= self.no_face_seconds:
                # Exceeded no_face_seconds: trigger NO_FACE state
                if not self.no_face_active:
                    self.no_face_active = True
                    timestamp = get_timestamp_str()
                    start_msg = f"DRIVER NOT VISIBLE: No face detected for {self.no_face_duration:.1f}s"
                    events.append({
                        "type": "NO_FACE_START",
                        "message": start_msg,
                        "time": get_time_str()
                    })
                    if self.log_to_console:
                        print(
                            f"[{timestamp}] [WARNING] NO_FACE_START: Driver not visible for {self.no_face_duration:.1f}s",
                            flush=True
                        )
                    # Play initial WARNING double-beep
                    self.audio_manager.play_warning()
                    self.last_no_face_beep_time = current_time

                # Check escalation to full siren vs repeating warning double-beep
                if self.no_face_duration >= self.no_face_alarm_seconds:
                    if not self.no_face_alarm_active:
                        self.no_face_alarm_active = True
                        timestamp = get_timestamp_str()
                        alarm_msg = (
                            f"DRIVER NOT VISIBLE ESCALATION: Siren alarm activated "
                            f"(no face for {self.no_face_duration:.1f}s)"
                        )
                        events.append({
                            "type": "NO_FACE_ALARM",
                            "message": alarm_msg,
                            "time": get_time_str()
                        })
                        if self.log_to_console:
                            print(
                                f"[{timestamp}] [ALERT]   NO_FACE_ALARM: Siren activated "
                                f"(driver missing for {self.no_face_duration:.1f}s)",
                                flush=True
                            )
                else:
                    # While missing (< alarm threshold), repeat warning beep every 5 seconds
                    if self.last_no_face_beep_time is not None and (current_time - self.last_no_face_beep_time) >= 5.0:
                        self.audio_manager.play_warning()
                        self.last_no_face_beep_time = current_time

        # Determine effective threshold
        effective_threshold = self.ear_threshold if self.ear_threshold is not None else self.fallback_ear_threshold

        # Eye Closure Evaluation
        if face_detected and not self.is_calibrating and screen_left_ear is not None and screen_right_ear is not None:
            screen_left_closed = screen_left_ear < effective_threshold
            screen_right_closed = screen_right_ear < effective_threshold
            both_closed = max(screen_left_ear, screen_right_ear) < effective_threshold
        else:
            screen_left_closed = False
            screen_right_closed = False
            both_closed = False

        # Yawn Detection Evaluation
        if face_detected and not self.is_calibrating and frame_mar is not None:
            if frame_mar >= self.mar_threshold:
                if self.yawn_start_time is None:
                    self.yawn_start_time = current_time
                self.yawn_duration = current_time - self.yawn_start_time

                if self.yawn_duration >= self.yawn_seconds and not self.is_yawning:
                    self.is_yawning = True
                    self.yawn_count += 1
                    timestamp = get_timestamp_str()
                    yawn_msg = f"Yawn #{self.yawn_count} detected (MAR: {frame_mar:.3f} for {self.yawn_duration:.2f}s)"
                    events.append({
                        "type": "YAWN",
                        "message": yawn_msg,
                        "time": get_time_str()
                    })

                    if self.log_to_console:
                        print(
                            f"[{timestamp}] [YAWN] Yawn #{self.yawn_count} detected! "
                            f"(MAR: {frame_mar:.3f} >= {self.mar_threshold:.2f} for {self.yawn_duration:.2f}s)",
                            flush=True
                        )

                    # Fatigue Warning Check
                    self.yawn_timestamps.append(current_time)
                    cutoff_time = current_time - self.yawn_warning_window
                    self.yawn_timestamps = [t for t in self.yawn_timestamps if t >= cutoff_time]
                    if len(self.yawn_timestamps) >= self.yawn_warning_count:
                        self.fatigue_warning_active = True
                        self.fatigue_warning_start_time = current_time
                        self.audio_manager.play_warning()
                        fatigue_msg = (
                            f"FATIGUE WARNING: {len(self.yawn_timestamps)} yawns "
                            f"detected within {self.yawn_warning_window:.0f}s! Take a break."
                        )
                        events.append({
                            "type": "FATIGUE_WARNING",
                            "message": fatigue_msg,
                            "time": get_time_str()
                        })

                        if self.log_to_console:
                            print(f"[{timestamp}] [WARNING] {fatigue_msg}", flush=True)

                        # Reset window
                        self.yawn_timestamps.clear()
            else:
                self.yawn_start_time = None
                self.yawn_duration = 0.0
                self.is_yawning = False
        else:
            self.yawn_start_time = None
            self.yawn_duration = 0.0
            self.is_yawning = False

        # Closure Counters
        if face_detected and not self.is_calibrating and both_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time
            self.consecutive_closed_frames += 1
            self.closed_duration = current_time - self.closed_start_time
        else:
            self.consecutive_closed_frames = 0
            self.closed_start_time = None
            self.closed_duration = 0.0

        # Drowsy State Decision
        self.drowsy_flag = (self.closed_duration >= self.threshold_seconds) and (self.consecutive_closed_frames >= dynamic_frame_threshold)

        # Audio ALARM Control (either Drowsy Eyes or Driver Not Visible Siren Escalation)
        if self.drowsy_flag or self.no_face_alarm_active:
            self.audio_manager.start_alarm()
        else:
            self.audio_manager.stop_alarm()

        # Drowsy State Transition Logging & Events
        if self.drowsy_flag != self.prev_drowsy_flag:
            timestamp = get_timestamp_str()
            if self.drowsy_flag:
                drowsy_msg = f"DROWSY ALERT: Both eyes closed for {self.closed_duration:.2f}s (EAR: {frame_ear:.3f})"
                events.append({
                    "type": "DROWSY_START",
                    "message": drowsy_msg,
                    "time": get_time_str()
                })
                if self.log_to_console:
                    print(
                        f"[{timestamp}] [ALERT] DROWSY state changed to: TRUE "
                        f"(Both eyes closed for {self.closed_duration:.2f}s | {self.consecutive_closed_frames} frames @ {self.fps_smooth:.1f} FPS | "
                        f"L: {screen_left_ear:.3f}, R: {screen_right_ear:.3f}, max: {max(screen_left_ear, screen_right_ear):.3f} < {effective_threshold:.3f})",
                        flush=True
                    )
            else:
                wake_msg = f"Eyes reopened. Driver awake (EAR: {frame_ear:.3f})"
                events.append({
                    "type": "DROWSY_STOP",
                    "message": wake_msg,
                    "time": get_time_str()
                })
                if self.log_to_console:
                    print(
                        f"[{timestamp}] [INFO]  DROWSY state changed to: FALSE "
                        f"(Driver eyes reopened | L: {screen_left_ear:.3f}, R: {screen_right_ear:.3f}, avg: {frame_ear:.3f})",
                        flush=True
                    )
            self.prev_drowsy_flag = self.drowsy_flag

        # Determine System Status Label & Colors
        if self.no_face_active:
            status_key = "NO_FACE"
            status_str = "STATUS: DRIVER NOT VISIBLE"
            status_color = (0, 215, 255)  # Amber (BGR)
        elif self.is_calibrating:
            status_key = "CALIBRATING"
            status_str = "STATUS: CALIBRATING... (Eyes open, mouth closed)"
            status_color = (0, 215, 255)  # Gold/Amber
        elif self.drowsy_flag:
            status_key = "DROWSY"
            status_str = "STATUS: *** DROWSY ALERT ***"
            status_color = (0, 0, 255)  # Bright Red
        elif self.fatigue_warning_active:
            status_key = "FATIGUE"
            status_str = "STATUS: *** FATIGUE WARNING: TAKE A BREAK ***"
            status_color = (0, 140, 255)  # Dark Orange
        elif self.is_yawning:
            status_key = "YAWNING"
            status_str = f"STATUS: *** YAWNING DETECTED (#{self.yawn_count}) ***"
            status_color = (0, 165, 255)  # Orange
        elif both_closed:
            status_key = "BOTH_EYES_CLOSED"
            status_str = "STATUS: Both Eyes Closed"
            status_color = (0, 165, 255)  # Orange
        elif screen_left_closed or screen_right_closed:
            status_key = "ONE_EYE_CLOSED"
            status_str = f"STATUS: One Eye Closed ({'Left' if screen_left_closed else 'Right'})"
            status_color = (0, 215, 255)  # Amber
        else:
            status_key = "AWAKE"
            status_str = "STATUS: Driver Awake"
            status_color = (0, 255, 0)  # Bright Green

        # Build EngineState snapshot
        state = EngineState(
            status=status_key,
            status_text=status_str,
            status_color=status_color,
            face_detected=face_detected,
            no_face_duration=self.no_face_duration,
            no_face_alarm_active=self.no_face_alarm_active,
            is_calibrating=self.is_calibrating,
            calibration_progress=calib_progress,
            calibration_remaining=calib_remaining,
            baseline_ear=self.baseline_ear,
            baseline_mar=self.baseline_mar,
            effective_ear_threshold=effective_threshold,
            frame_ear=frame_ear,
            left_ear=screen_left_ear,
            right_ear=screen_right_ear,
            frame_mar=frame_mar,
            yawn_count=self.yawn_count,
            is_yawning=self.is_yawning,
            yawn_duration=self.yawn_duration,
            drowsy_flag=self.drowsy_flag,
            closed_duration=self.closed_duration,
            consecutive_closed_frames=self.consecutive_closed_frames,
            fatigue_warning_flag=self.fatigue_warning_active,
            fps=self.fps_smooth,
            sound_status=self.audio_manager.get_status_str(),
            is_muted=self.audio_manager.is_muted,
            volume=self.audio_manager.volume,
            events=events
        )

        # -------------------------------------------------------------
        # Optional Visual Overlays on Frame
        # -------------------------------------------------------------
        if self.draw_overlays:
            # Draw Eye Bounding Boxes & Labels
            if face_detected and screen_left_bbox is not None and screen_right_bbox is not None:
                if self.is_calibrating:
                    left_color = (0, 215, 255)
                    right_color = (0, 215, 255)
                    left_label = f"L: {screen_left_ear:.2f}"
                    right_label = f"R: {screen_right_ear:.2f}"
                else:
                    left_color = (0, 0, 255) if screen_left_closed else (0, 255, 0)
                    right_color = (0, 0, 255) if screen_right_closed else (0, 255, 0)
                    left_label = f"L: {'CLOSED' if screen_left_closed else 'OPEN'} ({screen_left_ear:.2f})"
                    right_label = f"R: {'CLOSED' if screen_right_closed else 'OPEN'} ({screen_right_ear:.2f})"

                draw_styled_bbox(frame, screen_left_bbox, left_label, color=left_color, thickness=2)
                draw_styled_bbox(frame, screen_right_bbox, right_label, color=right_color, thickness=2)

            # Draw Mouth Bounding Box & MAR Label
            if face_detected and mouth_bbox is not None and frame_mar is not None:
                if self.is_calibrating:
                    m_color = (0, 215, 255)
                    m_label = f"Mouth: {frame_mar:.2f}"
                elif self.is_yawning:
                    m_color = (0, 69, 255)
                    m_label = f"YAWNING #{self.yawn_count} ({frame_mar:.2f})"
                elif frame_mar >= self.mar_threshold:
                    m_color = (0, 165, 255)
                    m_label = f"Mouth: {self.yawn_duration:.1f}s ({frame_mar:.2f})"
                else:
                    m_color = (0, 200, 255)
                    m_label = f"Mouth ({frame_mar:.2f})"

                draw_styled_bbox(frame, mouth_bbox, m_label, color=m_color, thickness=2)

            # 1. Top HUD Header Bar
            header_color = (0, 0, 180) if (self.drowsy_flag or self.no_face_alarm_active) else (25, 25, 25)
            cv2.rectangle(frame, (0, 0), (w, 65), header_color, -1)
            cv2.line(frame, (0, 65), (w, 65), (70, 70, 70), 1)

            cv2.putText(
                frame,
                status_str,
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                status_color,
                2,
                cv2.LINE_AA
            )

            if self.no_face_active:
                if self.no_face_alarm_active:
                    nf_sub = f"DRIVER NOT VISIBLE: SIREN ACTIVE (Missing for {self.no_face_duration:.1f}s >= {self.no_face_alarm_seconds:.1f}s)"
                else:
                    nf_sub = f"DRIVER NOT VISIBLE: Warning beeps active (Missing for {self.no_face_duration:.1f}s >= {self.no_face_seconds:.1f}s)"
                cv2.putText(
                    frame,
                    nf_sub,
                    (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA
                )
            elif face_detected and frame_ear is not None and screen_left_ear is not None and screen_right_ear is not None and frame_mar is not None:
                if self.is_calibrating:
                    ear_info = (
                        f"EAR: {frame_ear:.3f} (L: {screen_left_ear:.2f}, R: {screen_right_ear:.2f})  |  "
                        f"MAR: {frame_mar:.3f}  [Measuring Baselines...]"
                    )
                else:
                    ear_info = (
                        f"EAR: {frame_ear:.3f} (L:{screen_left_ear:.2f}, R:{screen_right_ear:.2f} | Thresh:{effective_threshold:.3f})  |  "
                        f"MAR: {frame_mar:.3f} (Thresh:{self.mar_threshold:.2f})  |  "
                        f"Yawns: {self.yawn_count}"
                    )

                cv2.putText(
                    frame,
                    ear_info,
                    (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

            # FPS in top-right
            fps_str = f"FPS: {self.fps_smooth:4.1f} (Threshold: {dynamic_frame_threshold} f)"
            (tw_fps, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(
                frame,
                fps_str,
                (w - tw_fps - 15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # SOUND: ON/MUTED/OFF status display in header
            if self.no_sound or not self.audio_manager.available:
                sound_str = "SOUND: OFF"
                sound_color = (130, 130, 130)
            elif self.audio_manager.is_muted:
                sound_str = "SOUND: MUTED [m]"
                sound_color = (0, 140, 255)
            else:
                sound_str = f"SOUND: ON ({int(self.audio_manager.volume * 100)}%)"
                sound_color = (0, 255, 0)

            (tw_snd, _), _ = cv2.getTextSize(sound_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(
                frame,
                sound_str,
                (w - tw_snd - 15, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                sound_color,
                1,
                cv2.LINE_AA
            )

            # 2. Startup Calibration Card
            if self.is_calibrating and not self.no_face_active:
                draw_calibration_card(frame, calib_elapsed, self.calibration_seconds, face_detected, frame_ear, frame_mar)

            # 3. Bottom Metric & Progress Bar
            bar_y = h - 45
            cv2.rectangle(frame, (0, bar_y - 10), (w, h), (20, 20, 20), -1)

            if self.no_face_active:
                counter_str = (
                    f"DRIVER NOT VISIBLE: Face absent for {self.no_face_duration:.1f}s  |  "
                    f"(Warning: {self.no_face_seconds:.1f}s | Siren: {self.no_face_alarm_seconds:.1f}s)"
                )
            elif self.is_calibrating:
                rem = max(0.0, self.calibration_seconds - calib_elapsed)
                counter_str = f"CALIBRATION PHASE: Eyes open, mouth closed ({rem:.1f}s remaining)"
            else:
                ear_str = f"{frame_ear:.3f}" if frame_ear is not None else "---"
                thresh_str = f"{effective_threshold:.3f}"
                mar_str = f"{frame_mar:.3f}" if frame_mar is not None else "---"
                counter_str = (
                    f"Closed: {self.consecutive_closed_frames:2d} / {dynamic_frame_threshold:2d} f ({self.closed_duration:3.1f}s)  |  "
                    f"Yawns: {self.yawn_count:2d} (MAR: {mar_str})  |  "
                    f"EAR: {ear_str} (Thresh: {thresh_str})"
                )

            cv2.putText(
                frame,
                counter_str,
                (15, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 215, 255) if (self.is_calibrating or self.consecutive_closed_frames > 0 or self.is_yawning or self.no_face_active) else (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # Progress Bar fill (toward drowsy alert duration)
            if not self.is_calibrating and not self.no_face_active:
                progress = min(1.0, self.closed_duration / self.threshold_seconds) if self.threshold_seconds > 0 else 0.0
                progress_bar_x = w - 220
                progress_bar_w = 200
                progress_bar_h = 14
                cv2.rectangle(
                    frame,
                    (progress_bar_x, bar_y + 5),
                    (progress_bar_x + progress_bar_w, bar_y + 5 + progress_bar_h),
                    (60, 60, 60),
                    1
                )

                fill_w = int(progress_bar_w * progress)
                fill_color = (0, 0, 255) if self.drowsy_flag else ((0, 165, 255) if progress > 0.5 else (0, 255, 0))
                if fill_w > 0:
                    cv2.rectangle(
                        frame,
                        (progress_bar_x + 1, bar_y + 6),
                        (progress_bar_x + fill_w, bar_y + 4 + progress_bar_h),
                        fill_color,
                        -1
                    )

            # 4. PIP Previews for Left and Right Eyes
            if screen_left_crop is not None and screen_right_crop is not None and screen_left_ear is not None and screen_right_ear is not None:
                pip_x = w - 105
                l_pip_color = (0, 0, 255) if screen_left_closed else (0, 255, 0)
                r_pip_color = (0, 0, 255) if screen_right_closed else (0, 255, 0)

                cnn_l_label = f"|{cnn_left_score:.2f}" if (self.show_cnn and cnn_left_score is not None) else ""
                cnn_r_label = f"|{cnn_right_score:.2f}" if (self.show_cnn and cnn_right_score is not None) else ""
                overlay_crop_pip(frame, screen_left_crop, pip_x, 75, f"L ({screen_left_ear:.2f}{cnn_l_label})", border_color=l_pip_color)
                overlay_crop_pip(frame, screen_right_crop, pip_x, 185, f"R ({screen_right_ear:.2f}{cnn_r_label})", border_color=r_pip_color)

            # 5. Alert Banners
            if self.drowsy_flag:
                draw_drowsy_banner(frame, self.closed_duration, frame_ear)
            elif self.no_face_alarm_active:
                draw_no_face_alarm_banner(frame, self.no_face_duration)
            elif self.fatigue_warning_active:
                draw_fatigue_warning_banner(frame)

        return frame, state

    def close(self):
        """Release audio and vision resources."""
        self.audio_manager.close()
        self.detector.close()
