"""
Audio Alert Test Script
Tests the ALARM siren (2 seconds) and WARNING double-beep (once) without requiring camera or vision models.
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_alerts import AudioManager


def main():
    parser = argparse.ArgumentParser(description="Test synthesized audio alerts without camera.")
    parser.add_argument("--volume", type=float, default=1.0, help="Test playback volume 0.0 to 1.0 (default: 1.0).")
    args = parser.parse_args()

    print("=" * 65)
    print("   AUDIO ALERT TEST - DROWSINESS DETECTION SYSTEM")
    print("=" * 65)
    print(f"[*] Testing with volume: {args.volume:.2f}")

    audio = AudioManager(no_sound=False, volume=args.volume)
    if not audio.available:
        print("[!] Audio device unavailable or pygame mixer failed. Cannot perform audio test.")
        sys.exit(1)

    try:
        print("\n[*] 1. Playing ALARM siren (loud, harsh two-tone 900Hz / 1400Hz) for 2.0s...")
        audio.start_alarm()
        time.sleep(2.0)
        audio.stop_alarm()
        print("[+] ALARM stopped successfully.")

        print("\n[*] Pausing 0.5s before warning beep...")
        time.sleep(0.5)

        print("[*] 2. Playing WARNING double beep (short 700Hz double beep) once...")
        audio.play_warning()
        time.sleep(0.6)  # Allow 0.40s double beep to complete
        print("[+] WARNING finished.")

        print("\n[+] All audio alert tests completed successfully.")
    finally:
        audio.close()


if __name__ == "__main__":
    main()
