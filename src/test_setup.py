"""
Webcam & Environment Verification Script
Drowsiness Detection System
"""

import sys
import cv2


def main():
    print("[*] Testing webcam connection...", flush=True)

    # Attempt to open default webcam (index 0)
    # Using cv2.CAP_DSHOW on Windows often speeds up camera initialization
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Fallback to default backend if CAP_DSHOW fails to initialize
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] ERROR: Could not open webcam.", flush=True)
        print("[!] Troubleshooting tips for Windows:", flush=True)
        print("    1. Ensure no other application (Zoom, Teams, Camera app) is using the webcam.", flush=True)
        print("    2. Check Windows Settings -> Privacy & security -> Camera -> Allow desktop apps access.", flush=True)
        print("    3. If you have an external USB webcam, verify it is connected and try index 1.", flush=True)
        sys.exit(1)

    print("Setup OK", flush=True)
    print("[*] Webcam opened successfully.", flush=True)
    print("[*] Focus the video window and press 'q' to quit.", flush=True)

    window_name = "Webcam Test - Drowsiness Detection (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from webcam.")
                break

            # Add subtle status text on video
            cv2.putText(
                frame,
                "Setup OK - Press 'q' to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)

            # Wait for 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[*] 'q' pressed. Exiting cleanly...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Camera released and windows closed. Ready for next step!")


if __name__ == "__main__":
    main()
