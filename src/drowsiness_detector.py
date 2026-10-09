"""
Real-Time Eye & Yawn Drowsiness Detection System (CLI Entrypoint)
Drowsiness Detection System for Indian Truck Drivers

Runs the DrowsinessEngine on live webcam feed.
"""

import sys
import argparse
from pathlib import Path
import cv2

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import DrowsinessEngine


def main():
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness & Yawn Detection System.")
    parser.add_argument("--threshold-seconds", type=float, default=1.5,
                        help="Continuous eye-closure duration in seconds before triggering alert (default: 1.5s).")
    parser.add_argument("--calibration-seconds", type=float, default=3.0,
                        help="Duration of startup calibration phase in seconds (default: 3.0s).")
    parser.add_argument("--ear-ratio", type=float, default=0.68,
                        help="Fraction of baseline open EAR used as closed threshold (default: 0.68).")
    parser.add_argument("--mar-threshold", type=float, default=0.60,
                        help="Mouth Aspect Ratio threshold to register a yawn opening (default: 0.60).")
    parser.add_argument("--yawn-seconds", type=float, default=1.2,
                        help="Continuous mouth opening duration in seconds to count as a yawn (default: 1.2s).")
    parser.add_argument("--yawn-warning-count", type=int, default=3,
                        help="Number of yawns within time window to trigger fatigue warning (default: 3).")
    parser.add_argument("--yawn-warning-window", type=float, default=600.0,
                        help="Time window in seconds for fatigue yawn count (default: 600.0s).")
    parser.add_argument("--no-sound", action="store_true", default=False,
                        help="Disable all audio alerts.")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Audio alert volume from 0.0 to 1.0 (default: 1.0).")
    parser.add_argument("--show-cnn", action="store_true", default=False,
                        help="Load and run CNN eye model for observation/display (default: False).")
    parser.add_argument("--eye-threshold", type=float, default=0.5,
                        help="Confidence threshold for CNN open eye classification [observation only] (default: 0.50).")
    parser.add_argument("--model-path", type=str, default="models/eye_model.keras",
                        help="Path to trained eye model (default: models/eye_model.keras).")
    parser.add_argument("--no-face-seconds", type=float, default=3.0,
                        help="Continuous duration in seconds with no face detected before warning (default: 3.0s).")
    parser.add_argument("--no-face-alarm-seconds", type=float, default=8.0,
                        help="Continuous duration in seconds with no face detected before siren alarm escalation (default: 8.0s).")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Webcam device index (default: 0).")

    args = parser.parse_args()

    print("=" * 65, flush=True)
    print("   REAL-TIME DROWSINESS & YAWN DETECTION SYSTEM (EAR + MAR)", flush=True)
    print("=" * 65, flush=True)
    print(f"[*] Eye Decision Signal: Eye Aspect Ratio (EAR) from MediaPipe landmarks", flush=True)
    print(f"[*] Yawn Decision Signal: Mouth Aspect Ratio (MAR) from MediaPipe inner lips", flush=True)
    print(f"[*] Calibration Phase  : {args.calibration_seconds:.1f}s at startup (eyes open, mouth closed)", flush=True)
    print(f"[*] Eye Closed Thresh  : {int(args.ear_ratio * 100)}% of calibrated baseline (--ear-ratio: {args.ear_ratio:.2f})", flush=True)
    print(f"[*] Drowsy Eye Time    : {args.threshold_seconds:.1f}s continuous closure", flush=True)
    print(f"[*] Yawn Detection     : MAR >= {args.mar_threshold:.2f} for >= {args.yawn_seconds:.1f}s", flush=True)
    print(f"[*] Fatigue Warning    : {args.yawn_warning_count} yawns within {args.yawn_warning_window:.0f}s window", flush=True)
    print(f"[*] No Face Warning    : Warning beep after {args.no_face_seconds:.1f}s missing (repeats every 5s)", flush=True)
    print(f"[*] No Face Alarm      : Siren escalation after {args.no_face_alarm_seconds:.1f}s missing", flush=True)
    print(f"[*] Audio Alerts       : {'DISABLED (--no-sound)' if args.no_sound else f'ENABLED (Volume: {args.volume:.2f})'}", flush=True)
    print(f"[*] Mirror Mode        : ENABLED (horizontal flip before landmark tracking)", flush=True)
    print(f"[*] CNN Eye Model      : {'ENABLED (--show-cnn)' if args.show_cnn else 'DISABLED (Pure Landmark mode)'}", flush=True)
    print(f"[*] Controls           : Press 'q' to quit, 'm' to mute/unmute audio\n", flush=True)

    # Initialize Core Engine
    print("[*] Initializing DrowsinessEngine...", flush=True)
    engine = DrowsinessEngine(
        threshold_seconds=args.threshold_seconds,
        calibration_seconds=args.calibration_seconds,
        ear_ratio=args.ear_ratio,
        mar_threshold=args.mar_threshold,
        yawn_seconds=args.yawn_seconds,
        yawn_warning_count=args.yawn_warning_count,
        yawn_warning_window=args.yawn_warning_window,
        no_sound=args.no_sound,
        volume=args.volume,
        show_cnn=args.show_cnn,
        eye_threshold=args.eye_threshold,
        model_path=args.model_path,
        mirror=True,
        draw_overlays=True,
        log_to_console=True,
        no_face_seconds=args.no_face_seconds,
        no_face_alarm_seconds=args.no_face_alarm_seconds
    )
    print("[+] DrowsinessEngine ready.", flush=True)

    # Open Webcam
    print(f"[*] Opening webcam index {args.camera_index}...", flush=True)
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("[!] ERROR: Could not open webcam. Check device permissions or index.", flush=True)
        sys.exit(1)

    print("Setup OK", flush=True)
    print("[*] Camera connected. Real-time inference running.", flush=True)
    print("[*] Calibration will begin as soon as your face is in view.", flush=True)
    print("[*] Press 'q' to exit, 'm' to toggle audio mute.\n", flush=True)

    window_name = "Drowsiness & Yawn Detection - Phase 4"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame from webcam.", flush=True)
                break

            annotated_frame, state = engine.process_frame(frame)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n[*] 'q' key pressed. Closing cleanly...", flush=True)
                break
            elif key in (ord('m'), ord('M')):
                muted_state = engine.toggle_mute()
                print(f"[*] Audio {'MUTED' if muted_state else 'UNMUTED'}", flush=True)

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()
        print("[*] Resources released. Drowsiness detection terminated cleanly.", flush=True)


if __name__ == "__main__":
    main()
