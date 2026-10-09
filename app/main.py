"""
PySide6 Desktop Application for Driver Drowsiness & Yawn Detection System.

Run Instructions:
    .\\venv\\Scripts\\python.exe app/main.py

    Or from an active virtual environment:
    python app/main.py

Requirements:
    - Camera / Webcam connected
    - PySide6, OpenCV, MediaPipe, NumPy, Pygame installed (via requirements.txt)
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QPlainTextEdit,
    QFrame,
    QSplitter,
    QSizePolicy
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.landmarks import (
    get_bounding_box,
    crop_region,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    MOUTH_INDICES
)
from src.engine import DrowsinessEngine, EngineState, overlay_crop_pip, draw_styled_bbox


class VideoWorker(QThread):
    """
    Background worker thread handling real-time webcam capture and detection inference.
    Prevents UI freezing by offloading all OpenCV and MediaPipe processing.
    """
    frame_ready = Signal(np.ndarray, object)
    log_event = Signal(str, str, str)  # (timestamp, event_type, message)
    error_occurred = Signal(str)
    worker_stopped = Signal()

    def __init__(
        self,
        camera_index: int = 0,
        threshold_seconds: float = 1.5,
        calibration_seconds: float = 3.0,
        ear_ratio: float = 0.68,
        mar_threshold: float = 0.60,
        yawn_seconds: float = 1.2,
        volume: float = 1.0,
        no_sound: bool = False,
        no_face_seconds: float = 3.0,
        no_face_alarm_seconds: float = 8.0
    ):
        super().__init__()
        self.camera_index = camera_index
        self.threshold_seconds = threshold_seconds
        self.calibration_seconds = calibration_seconds
        self.ear_ratio = ear_ratio
        self.mar_threshold = mar_threshold
        self.yawn_seconds = yawn_seconds
        self.volume = volume
        self.no_sound = no_sound
        self.no_face_seconds = no_face_seconds
        self.no_face_alarm_seconds = no_face_alarm_seconds

        self._is_running = False
        self.engine = None
        self.cap = None
        self.cached_landmarks = None

    def run(self):
        """Main camera acquisition and engine processing loop."""
        self._is_running = True

        # 1. Initialize Detection Engine with draw_overlays=False (clean frame for Qt rendering)
        try:
            self.engine = DrowsinessEngine(
                threshold_seconds=self.threshold_seconds,
                calibration_seconds=self.calibration_seconds,
                ear_ratio=self.ear_ratio,
                mar_threshold=self.mar_threshold,
                yawn_seconds=self.yawn_seconds,
                no_sound=self.no_sound,
                volume=self.volume,
                mirror=True,
                draw_overlays=False,
                log_to_console=False,
                no_face_seconds=self.no_face_seconds,
                no_face_alarm_seconds=self.no_face_alarm_seconds
            )
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize DrowsinessEngine: {e}")
            self._is_running = False
            return

        # Intercept MediaPipe landmarker output to capture face landmarks for clean Qt drawing
        self.cached_landmarks = None
        orig_detector_process = self.engine.detector.process

        def intercept_detector_process(img):
            res = orig_detector_process(img)
            self.cached_landmarks = res[0] if (res and len(res) > 0) else None
            return res

        self.engine.detector.process = intercept_detector_process

        # 2. Open Video Capture
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open camera index {self.camera_index}.")
            if self.engine:
                self.engine.close()
            self._is_running = False
            return

        self.log_event.emit(
            datetime.now().strftime("%H:%M:%S"),
            "SYSTEM",
            f"Camera {self.camera_index} connected. Inference active."
        )

        while self._is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process frame through engine (clean mirrored frame + EngineState)
            try:
                clean_frame, state = self.engine.process_frame(frame)
                rendered_frame = self._render_qt_overlay(clean_frame, state, self.cached_landmarks)
            except Exception as e:
                self.error_occurred.emit(f"Engine processing error: {e}")
                break

            # Forward newly triggered events to UI
            for evt in state.events:
                self.log_event.emit(evt.get("time", ""), evt.get("type", "INFO"), evt.get("message", ""))

            # Emit frame for UI presentation
            self.frame_ready.emit(rendered_frame, state)

            # Minimal yield to keep UI responsive
            self.msleep(1)

        # Cleanup on loop exit
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.engine:
            self.engine.close()

        self.worker_stopped.emit()

    def _render_qt_overlay(self, frame: np.ndarray, state: EngineState, landmarks) -> np.ndarray:
        """
        Render only eye/mouth boxes and small eye preview crops on the frame for the Qt window.
        Omits top header text and bottom metrics bar (already presented in side panels).
        Prevents L/R label overlap by positioning compact labels ("L 0.31") above each box.
        """
        if not state.face_detected or landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # 1. Compute Eye Bounding Boxes
        bbox_1 = get_bounding_box(landmarks, LEFT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)
        bbox_2 = get_bounding_box(landmarks, RIGHT_EYE_INDICES, w, h, margin_x=0.30, margin_y=0.40, make_square=True)

        # Screen-relative eye designation: box with smaller x center is Screen "L"
        cx_1 = (bbox_1[0] + bbox_1[2]) / 2.0
        cx_2 = (bbox_2[0] + bbox_2[2]) / 2.0

        if cx_1 < cx_2:
            screen_left_bbox = bbox_1
            screen_right_bbox = bbox_2
        else:
            screen_left_bbox = bbox_2
            screen_right_bbox = bbox_1

        # 2. Compute Mouth Bounding Box
        mouth_bbox = get_bounding_box(landmarks, MOUTH_INDICES, w, h, margin_x=0.25, margin_y=0.30, make_square=True)

        # 3. Crop 96x96 Eye Patches for Preview PIPs
        screen_left_crop = crop_region(frame, screen_left_bbox, target_size=(96, 96))
        screen_right_crop = crop_region(frame, screen_right_bbox, target_size=(96, 96))

        # 4. Determine Eye Colors and Shortened Labels ("L 0.31", "R 0.31")
        if state.is_calibrating:
            left_color = (0, 215, 255)   # Amber
            right_color = (0, 215, 255)
        else:
            left_closed = (state.left_ear is not None) and (state.left_ear < state.effective_ear_threshold)
            right_closed = (state.right_ear is not None) and (state.right_ear < state.effective_ear_threshold)
            left_color = (0, 0, 255) if left_closed else (0, 255, 0)
            right_color = (0, 0, 255) if right_closed else (0, 255, 0)

        left_val_str = f"{state.left_ear:.2f}" if state.left_ear is not None else "--"
        right_val_str = f"{state.right_ear:.2f}" if state.right_ear is not None else "--"
        left_label = f"L {left_val_str}"
        right_label = f"R {right_val_str}"

        # Draw compact eye boxes with shortened labels directly above each box to prevent overlap
        def draw_compact_eye_box(img, bbox, label, color):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

            # Draw label above its own box without bleeding into adjacent eye
            label_y = max(y1 - 4, th + 4)
            badge_x2 = min(x2, x1 + tw + 6)
            cv2.rectangle(img, (x1, label_y - th - 3), (badge_x2, label_y + 3), color, -1)
            cv2.putText(img, label, (x1 + 3, label_y - 1), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        draw_compact_eye_box(frame, screen_left_bbox, left_label, left_color)
        draw_compact_eye_box(frame, screen_right_bbox, right_label, right_color)

        # 5. Draw Mouth Box
        mar_str = f"{state.frame_mar:.2f}" if state.frame_mar is not None else "--"
        if state.is_calibrating:
            m_color = (0, 215, 255)
            m_label = f"Mouth {mar_str}"
        elif state.is_yawning:
            m_color = (0, 69, 255)
            m_label = f"YAWN #{state.yawn_count} ({mar_str})"
        elif (state.frame_mar is not None and state.frame_mar >= self.mar_threshold):
            m_color = (0, 165, 255)
            m_label = f"Mouth {state.yawn_duration:.1f}s ({mar_str})"
        else:
            m_color = (0, 200, 255)
            m_label = f"Mouth {mar_str}"

        draw_styled_bbox(frame, mouth_bbox, m_label, color=m_color, thickness=2)

        # 6. Draw Small Eye Preview Crops (96x96) in Top-Right Corner
        if screen_left_crop is not None and screen_right_crop is not None:
            pip_x = w - 105
            overlay_crop_pip(frame, screen_left_crop, pip_x, 15, f"L {left_val_str}", border_color=left_color)
            overlay_crop_pip(frame, screen_right_crop, pip_x, 125, f"R {right_val_str}", border_color=right_color)

        return frame

    def stop(self):
        """Signal thread to terminate and wait for clean shutdown."""
        self._is_running = False
        self.wait(2500)

    # Dynamic thread-safe setters called from UI
    def recalibrate(self):
        if self.engine:
            self.engine.recalibrate()
            self.log_event.emit(
                datetime.now().strftime("%H:%M:%S"),
                "CALIBRATION",
                "Recalibration initiated: Look at camera with eyes open and mouth closed."
            )

    def set_threshold_seconds(self, seconds: float):
        self.threshold_seconds = float(seconds)
        if self.engine:
            self.engine.set_threshold_seconds(seconds)

    def set_mar_threshold(self, thresh: float):
        self.mar_threshold = float(thresh)
        if self.engine:
            self.engine.set_mar_threshold(thresh)

    def set_volume(self, volume: float):
        self.volume = float(volume)
        if self.engine:
            self.engine.set_volume(volume)

    def set_mute(self, muted: bool):
        if self.engine:
            self.engine.set_mute(muted)

    def toggle_mute(self) -> bool:
        if self.engine:
            return self.engine.toggle_mute()
        return False


class MainWindow(QMainWindow):
    """
    PySide6 Modern Dark-Themed Desktop UI for Driver Drowsiness Detection.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness & Fatigue Detection System")
        self.resize(1240, 800)
        self.setMinimumSize(1020, 680)

        self.worker = None
        self.is_muted = False

        self._apply_dark_theme()
        self._setup_ui()
        self._set_ui_stopped_state()

    def _apply_dark_theme(self):
        """Set a clean, professional dark palette with curated colors."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121418;
            }
            QWidget {
                color: #e5e7eb;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #1a1e24;
                border: 1px solid #2d3340;
                border-radius: 8px;
                margin-top: 22px;
                font-weight: bold;
                font-size: 13px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 3px 10px;
                color: #94a3b8;
            }
            QFrame#card {
                background-color: #1a1e24;
                border: 1px solid #2d3340;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #272d38;
                color: #f1f5f9;
                border: 1px solid #3b4252;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #333b49;
                border-color: #4c566a;
            }
            QPushButton:pressed {
                background-color: #1e232c;
            }
            QPushButton:disabled {
                background-color: #181b21;
                color: #4b5563;
                border-color: #232730;
            }
            QPushButton#btnStart {
                background-color: #065f46;
                border-color: #059669;
                color: #ffffff;
            }
            QPushButton#btnStart:hover {
                background-color: #047857;
            }
            QPushButton#btnStart:disabled {
                background-color: #132720;
                border-color: #1f3b32;
                color: #4b6357;
            }
            QPushButton#btnStop {
                background-color: #991b1b;
                border-color: #dc2626;
                color: #ffffff;
            }
            QPushButton#btnStop:hover {
                background-color: #b91c1c;
            }
            QPushButton#btnStop:disabled {
                background-color: #2d1818;
                border-color: #422222;
                color: #634343;
            }
            QPushButton#btnRecalib {
                background-color: #1e40af;
                border-color: #2563eb;
                color: #ffffff;
            }
            QPushButton#btnRecalib:hover {
                background-color: #1d4ed8;
            }
            QPushButton#btnMute {
                background-color: #374151;
                border-color: #4b5563;
                color: #ffffff;
            }
            QPushButton#btnMute:hover {
                background-color: #4b5563;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2d3340;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2563eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #93c5fd;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #121418;
                border: 1px solid #333b49;
                border-radius: 5px;
                padding: 4px 8px;
                color: #f1f5f9;
                font-weight: 600;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
            }
            QPlainTextEdit {
                background-color: #121418;
                border: 1px solid #2d3340;
                border-radius: 6px;
                color: #cbd5e1;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 6px;
            }
        """)

    def _setup_ui(self):
        """Construct the split layout: Video & Quick Metrics (Left), Controls & Status (Right)."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # =============================================================
        # Left Column: Live Video Feed & Secondary Gauge Bar
        # =============================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Video Frame Card
        video_card = QFrame()
        video_card.setObjectName("card")
        video_card_layout = QVBoxLayout(video_card)
        video_card_layout.setContentsMargins(8, 8, 8, 8)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: #0b0c0e; border-radius: 6px;")
        video_card_layout.addWidget(self.video_label)
        left_layout.addWidget(video_card, stretch=5)

        # Bottom Quick Metrics Bar
        metrics_card = QFrame()
        metrics_card.setObjectName("card")
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(16, 12, 16, 12)
        metrics_layout.setSpacing(24)

        # Metric 1: EAR
        self.val_ear = self._create_metric_widget("EAR (Eye Aspect)", "0.000", "#38bdf8")
        metrics_layout.addWidget(self.val_ear)

        # Metric 2: MAR
        self.val_mar = self._create_metric_widget("MAR (Mouth Aspect)", "0.000", "#fbbf24")
        metrics_layout.addWidget(self.val_mar)

        # Metric 3: Yawns
        self.val_yawns = self._create_metric_widget("Yawns Count", "0", "#f87171")
        metrics_layout.addWidget(self.val_yawns)

        # Metric 4: FPS
        self.val_fps = self._create_metric_widget("Processing FPS", "0.0", "#4ade80")
        metrics_layout.addWidget(self.val_fps)

        left_layout.addWidget(metrics_card, stretch=1)
        splitter.addWidget(left_widget)

        # =============================================================
        # Right Column: Big Status Banner, Controls, Sliders, Log
        # =============================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)

        # 1. Big Status Label Card
        self.status_card = QFrame()
        self.status_card.setObjectName("card")
        status_card_layout = QVBoxLayout(self.status_card)
        status_card_layout.setContentsMargins(16, 16, 16, 16)
        status_card_layout.setSpacing(6)

        status_header_lbl = QLabel("SYSTEM STATUS")
        status_header_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8; letter-spacing: 1px;")
        status_card_layout.addWidget(status_header_lbl)

        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.status_label.setMinimumHeight(54)
        self.status_label.setStyleSheet("""
            background-color: #1e293b;
            color: #94a3b8;
            border: 2px solid #334155;
            border-radius: 8px;
            padding: 8px 12px;
        """)
        status_card_layout.addWidget(self.status_label)

        self.status_sublabel = QLabel("Click 'Start Detection' to begin camera stream.")
        self.status_sublabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_sublabel.setStyleSheet("font-size: 12px; color: #94a3b8;")
        status_card_layout.addWidget(self.status_sublabel)

        right_layout.addWidget(self.status_card)

        # 2. Control Buttons
        btn_box = QGroupBox("Controls")
        btn_layout = QGridLayout(btn_box)
        btn_layout.setContentsMargins(12, 16, 12, 12)
        btn_layout.setSpacing(10)

        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self._start_detection)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.clicked.connect(self._stop_detection)

        self.btn_recalib = QPushButton("Recalibrate")
        self.btn_recalib.setObjectName("btnRecalib")
        self.btn_recalib.clicked.connect(self._recalibrate)

        self.btn_mute = QPushButton("Mute Audio")
        self.btn_mute.setObjectName("btnMute")
        self.btn_mute.clicked.connect(self._toggle_mute)

        btn_layout.addWidget(self.btn_start, 0, 0)
        btn_layout.addWidget(self.btn_stop, 0, 1)
        btn_layout.addWidget(self.btn_recalib, 1, 0)
        btn_layout.addWidget(self.btn_mute, 1, 1)
        right_layout.addWidget(btn_box)

        # 3. Tuning Sliders & Spinboxes
        tuning_box = QGroupBox("Sensitivity & Audio Tuning")
        tuning_layout = QVBoxLayout(tuning_box)
        tuning_layout.setContentsMargins(12, 16, 12, 14)
        tuning_layout.setSpacing(12)

        # Param 1: Eye Closed Duration
        tuning_layout.addLayout(self._create_slider_row(
            label="Eye Closed Trigger (sec):",
            min_val=0.5,
            max_val=4.0,
            step=0.1,
            default_val=1.5,
            on_change=self._on_threshold_changed,
            attr_name="spin_thresh",
            slider_name="slider_thresh"
        ))

        # Param 2: Yawn MAR Threshold
        tuning_layout.addLayout(self._create_slider_row(
            label="Yawn MAR Threshold:",
            min_val=0.30,
            max_val=1.00,
            step=0.05,
            default_val=0.60,
            on_change=self._on_mar_changed,
            attr_name="spin_mar",
            slider_name="slider_mar"
        ))

        # Param 3: Audio Volume
        tuning_layout.addLayout(self._create_volume_row(
            default_val=100,
            on_change=self._on_volume_changed
        ))

        right_layout.addWidget(tuning_box)

        # 4. Event Log
        log_box = QGroupBox("Event Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(12, 14, 12, 12)
        log_layout.setSpacing(8)

        log_hdr_layout = QHBoxLayout()
        log_desc_lbl = QLabel("Timestamped system events:")
        log_desc_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        log_hdr_layout.addWidget(log_desc_lbl)

        btn_clear_log = QPushButton("Clear")
        btn_clear_log.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        btn_clear_log.clicked.connect(self._clear_log)
        log_hdr_layout.addWidget(btn_clear_log)
        log_layout.addLayout(log_hdr_layout)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(250)
        log_layout.addWidget(self.log_text)

        right_layout.addWidget(log_box, stretch=2)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)

    # -------------------------------------------------------------
    # UI Helper Builders
    # -------------------------------------------------------------
    def _create_metric_widget(self, title: str, initial_val: str, color_hex: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        layout.addWidget(title_lbl)

        val_lbl = QLabel(initial_val)
        val_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        val_lbl.setStyleSheet(f"color: {color_hex};")
        layout.addWidget(val_lbl)

        # Store label reference on container
        container.val_label = val_lbl
        return container

    def _create_slider_row(self, label: str, min_val: float, max_val: float, step: float, default_val: float, on_change, attr_name: str, slider_name: str):
        row = QVBoxLayout()
        row.setSpacing(4)

        hdr = QHBoxLayout()
        hdr_lbl = QLabel(label)
        hdr_lbl.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        hdr.addWidget(hdr_lbl)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(2 if step < 0.1 else 1)
        spin.setValue(default_val)
        spin.setFixedWidth(75)
        hdr.addWidget(spin)
        row.addLayout(hdr)

        slider = QSlider(Qt.Orientation.Horizontal)
        # Convert float to int scale (x100)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default_val * 100))
        slider.setSingleStep(int(step * 100))
        row.addWidget(slider)

        # Synchronize slider & spinbox
        def on_slider_move(val):
            f_val = val / 100.0
            spin.blockSignals(True)
            spin.setValue(f_val)
            spin.blockSignals(False)
            on_change(f_val)

        def on_spin_change(val):
            slider.blockSignals(True)
            slider.setValue(int(val * 100))
            slider.blockSignals(False)
            on_change(val)

        slider.valueChanged.connect(on_slider_move)
        spin.valueChanged.connect(on_spin_change)

        setattr(self, attr_name, spin)
        setattr(self, slider_name, slider)
        return row

    def _create_volume_row(self, default_val: int, on_change):
        row = QVBoxLayout()
        row.setSpacing(4)

        hdr = QHBoxLayout()
        hdr_lbl = QLabel("Alert Volume:")
        hdr_lbl.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        hdr.addWidget(hdr_lbl)

        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setSingleStep(5)
        spin.setValue(default_val)
        spin.setSuffix("%")
        spin.setFixedWidth(75)
        hdr.addWidget(spin)
        row.addLayout(hdr)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(default_val)
        slider.setSingleStep(5)
        row.addWidget(slider)

        def on_slider_move(val):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            on_change(val)

        def on_spin_change(val):
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
            on_change(val)

        slider.valueChanged.connect(on_slider_move)
        spin.valueChanged.connect(on_spin_change)

        self.spin_volume = spin
        self.slider_volume = slider
        return row

    # -------------------------------------------------------------
    # State Management & Controls
    # -------------------------------------------------------------
    def _set_ui_stopped_state(self):
        """Configure UI widgets for stopped / idle state."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_recalib.setEnabled(False)

        self.status_label.setText("DETECTION STOPPED")
        self.status_label.setStyleSheet("""
            background-color: #1e293b;
            color: #94a3b8;
            border: 2px solid #334155;
            border-radius: 8px;
            padding: 8px 12px;
        """)
        self.status_sublabel.setText("Click 'Start' to activate camera and detection engine.")

        # Display placeholder in video frame
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor("#0f1217"))
        self.video_label.setPixmap(pixmap)

    def _start_detection(self):
        """Instantiate and start the video worker thread."""
        if self.worker is not None and self.worker.isRunning():
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_recalib.setEnabled(True)

        thresh_sec = self.spin_thresh.value()
        mar_thresh = self.spin_mar.value()
        vol_frac = self.spin_volume.value() / 100.0

        self.worker = VideoWorker(
            camera_index=0,
            threshold_seconds=thresh_sec,
            calibration_seconds=3.0,
            ear_ratio=0.68,
            mar_threshold=mar_thresh,
            yawn_seconds=1.2,
            volume=vol_frac,
            no_sound=False,
            no_face_seconds=3.0,
            no_face_alarm_seconds=8.0
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.log_event.connect(self._append_log)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.worker_stopped.connect(self._on_worker_stopped)
        self.worker.start()

        self._append_log(datetime.now().strftime("%H:%M:%S"), "SYSTEM", "Detection worker started.")

    def _stop_detection(self):
        """Stop worker and release camera and audio resources."""
        if self.worker is not None and self.worker.isRunning():
            self._append_log(datetime.now().strftime("%H:%M:%S"), "SYSTEM", "Stopping detection worker...")
            self.worker.stop()
            self._set_ui_stopped_state()

    def _recalibrate(self):
        """Trigger baseline recalibration."""
        if self.worker and self.worker.isRunning():
            self.worker.recalibrate()

    def _toggle_mute(self):
        """Toggle audio alert mute state."""
        self.is_muted = not self.is_muted
        if self.worker and self.worker.isRunning():
            self.worker.set_mute(self.is_muted)

        if self.is_muted:
            self.btn_mute.setText("Unmute Audio")
            self.btn_mute.setStyleSheet("""
                background-color: #78350f;
                border-color: #d97706;
                color: #fef3c7;
            """)
            self._append_log(datetime.now().strftime("%H:%M:%S"), "AUDIO", "Alert sounds muted.")
        else:
            self.btn_mute.setText("Mute Audio")
            self.btn_mute.setStyleSheet("")
            self._append_log(datetime.now().strftime("%H:%M:%S"), "AUDIO", "Alert sounds unmuted.")

    # -------------------------------------------------------------
    # Slider Callbacks
    # -------------------------------------------------------------
    def _on_threshold_changed(self, val: float):
        if self.worker and self.worker.isRunning():
            self.worker.set_threshold_seconds(val)

    def _on_mar_changed(self, val: float):
        if self.worker and self.worker.isRunning():
            self.worker.set_mar_threshold(val)

    def _on_volume_changed(self, val: int):
        frac = val / 100.0
        if self.worker and self.worker.isRunning():
            self.worker.set_volume(frac)

    # -------------------------------------------------------------
    # Frame & State Receiver Slot
    # -------------------------------------------------------------
    @Slot(np.ndarray, object)
    def _on_frame_ready(self, frame: np.ndarray, state: EngineState):
        """Update live video panel and UI metric indicators."""
        # 1. Update Video Display
        h, w, ch = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to match video_label bounds preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        # 2. Update Status Card
        self._update_status_display(state)

        # 3. Update Metrics Bar
        ear_text = f"{state.frame_ear:.3f}" if state.frame_ear is not None else "---"
        mar_text = f"{state.frame_mar:.3f}" if state.frame_mar is not None else "---"
        self.val_ear.val_label.setText(ear_text)
        self.val_mar.val_label.setText(mar_text)
        self.val_yawns.val_label.setText(str(state.yawn_count))
        self.val_fps.val_label.setText(f"{state.fps:.1f}")

    def _update_status_display(self, state: EngineState):
        """Update the large colored status banner based on the active detection state."""
        if state.status == "NO_FACE":
            self.status_label.setText("DRIVER NOT VISIBLE")
            if state.no_face_alarm_active:
                self.status_label.setStyleSheet("""
                    background-color: #7f1d1d;
                    color: #fef3c7;
                    border: 2px solid #ef4444;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-weight: bold;
                """)
                self.status_sublabel.setText(f"SIREN ACTIVE: Driver missing for {state.no_face_duration:.1f}s!")
            else:
                self.status_label.setStyleSheet("""
                    background-color: #78350f;
                    color: #fbbf24;
                    border: 2px solid #f59e0b;
                    border-radius: 8px;
                    padding: 8px 12px;
                """)
                self.status_sublabel.setText(f"Driver missing for {state.no_face_duration:.1f}s. Warning beeps active.")
        elif state.is_calibrating:
            self.status_label.setText("CALIBRATING BASELINES")
            self.status_label.setStyleSheet("""
                background-color: #0c4a6e;
                color: #38bdf8;
                border: 2px solid #0284c7;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            self.status_sublabel.setText(f"Look at camera with eyes open, mouth closed ({state.calibration_remaining:.1f}s left)")
        elif state.drowsy_flag:
            self.status_label.setText("*** DROWSY ALERT! WAKE UP! ***")
            self.status_label.setStyleSheet("""
                background-color: #7f1d1d;
                color: #ffffff;
                border: 2px solid #ef4444;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
            """)
            self.status_sublabel.setText(f"Eyes continuously closed for {state.closed_duration:.1f}s!")
        elif state.fatigue_warning_flag:
            self.status_label.setText("*** FATIGUE WARNING: TAKE A BREAK ***")
            self.status_label.setStyleSheet("""
                background-color: #78350f;
                color: #fef3c7;
                border: 2px solid #f59e0b;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            self.status_sublabel.setText(f"Frequent yawning detected ({state.yawn_count} total). Pull over safely.")
        elif state.is_yawning:
            self.status_label.setText(f"YAWNING DETECTED (#{state.yawn_count})")
            self.status_label.setStyleSheet("""
                background-color: #78350f;
                color: #fde047;
                border: 2px solid #eab308;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            self.status_sublabel.setText(f"Mouth open for {state.yawn_duration:.1f}s (MAR: {state.frame_mar:.2f})")
        elif state.status == "BOTH_EYES_CLOSED":
            self.status_label.setText("BOTH EYES CLOSED")
            self.status_label.setStyleSheet("""
                background-color: #451a03;
                color: #fb923c;
                border: 2px solid #f97316;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            self.status_sublabel.setText(f"Closed for {state.closed_duration:.1f}s / {self.spin_thresh.value():.1f}s trigger")
        elif state.status == "ONE_EYE_CLOSED":
            self.status_label.setText("ONE EYE CLOSED")
            self.status_label.setStyleSheet("""
                background-color: #3b280c;
                color: #fcd34d;
                border: 2px solid #d97706;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            self.status_sublabel.setText(f"L: {state.left_ear:.2f} | R: {state.right_ear:.2f}")
        else:
            self.status_label.setText("DRIVER AWAKE")
            self.status_label.setStyleSheet("""
                background-color: #064e3b;
                color: #34d399;
                border: 2px solid #10b981;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            base_str = f"(Base: {state.baseline_ear:.2f})" if state.baseline_ear else ""
            self.status_sublabel.setText(f"Normal eye and mouth aspect ratios {base_str}")

    # -------------------------------------------------------------
    # Logging & Error Slots
    # -------------------------------------------------------------
    @Slot(str, str, str)
    def _append_log(self, time_str: str, event_type: str, message: str):
        """Append color-styled message to event log."""
        color_map = {
            "SYSTEM": "#94a3b8",
            "CALIBRATION": "#38bdf8",
            "CALIBRATION_START": "#38bdf8",
            "CALIBRATION_DONE": "#4ade80",
            "DROWSY_START": "#ef4444",
            "DROWSY_STOP": "#22c55e",
            "NO_FACE_START": "#f59e0b",
            "NO_FACE_ALARM": "#ef4444",
            "NO_FACE_STOP": "#10b981",
            "YAWN": "#fbbf24",
            "FATIGUE_WARNING": "#f97316",
            "AUDIO": "#c084fc",
            "ERROR": "#f87171"
        }
        color = color_map.get(event_type, "#cbd5e1")
        html = f"<span style='color: #64748b;'>[{time_str}]</span> <span style='color: {color}; font-weight: bold;'>[{event_type}]</span> {message}"
        self.log_text.appendHtml(html)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _clear_log(self):
        self.log_text.clear()

    @Slot(str)
    def _on_worker_error(self, err_msg: str):
        self._append_log(datetime.now().strftime("%H:%M:%S"), "ERROR", err_msg)
        self._set_ui_stopped_state()

    @Slot()
    def _on_worker_stopped(self):
        self._set_ui_stopped_state()
        self._append_log(datetime.now().strftime("%H:%M:%S"), "SYSTEM", "Detection terminated cleanly.")

    # -------------------------------------------------------------
    # Shutdown Safety
    # -------------------------------------------------------------
    def closeEvent(self, event):
        """Ensure background worker and audio/video resources cleanly terminate on window close."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
