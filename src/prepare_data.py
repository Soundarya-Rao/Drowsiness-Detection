"""
Dataset Preparation and Organization Script
Drowsiness Detection System for Indian Truck Drivers

Organizes raw downloaded datasets into standardized project directories:
  - MRL Eye Dataset  -> data/eyes/open/ and data/eyes/closed/
  - YawnDD / Yawn    -> data/yawn/yawn/ and data/yawn/no_yawn/
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

# Supported image file extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EYES_OPEN_DIR = DATA_DIR / "eyes" / "open"
EYES_CLOSED_DIR = DATA_DIR / "eyes" / "closed"
YAWN_DIR = DATA_DIR / "yawn" / "yawn"
NO_YAWN_DIR = DATA_DIR / "yawn" / "no_yawn"


def ensure_target_directories():
    """Create target destination directories if they don't exist."""
    for d in [EYES_OPEN_DIR, EYES_CLOSED_DIR, YAWN_DIR, NO_YAWN_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    """Check if file has an image extension."""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def extract_zip_if_needed(archive_path: Path, extract_to: Path) -> Path:
    """Extract a zip archive to a destination folder if not already extracted."""
    extract_to.mkdir(parents=True, exist_ok=True)
    marker_file = extract_to / ".extracted_marker"
    if marker_file.exists():
        return extract_to

    print(f"[*] Extracting archive '{archive_path.name}' to '{extract_to}'...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    marker_file.touch()
    print(f"[+] Extraction complete for '{archive_path.name}'.")
    return extract_to


def safe_copy_image(src_file: Path, dest_dir: Path, prefix: str = ""):
    """Copy an image to destination directory, avoiding filename collisions."""
    dest_name = f"{prefix}_{src_file.name}" if prefix else src_file.name
    dest_path = dest_dir / dest_name

    # If collision occurs, append numerical suffix
    counter = 1
    while dest_path.exists():
        stem = f"{prefix}_{src_file.stem}" if prefix else src_file.stem
        dest_path = dest_dir / f"{stem}_{counter}{src_file.suffix}"
        counter += 1

    shutil.copy2(src_file, dest_path)


def classify_mrl_file(file_path: Path) -> str | None:
    """
    Determine whether an eye image is 'open' or 'closed'.
    Supports:
      1. Standard MRL filename format: s0001_00001_0_0_0_0_0_01.png
         Tokens: [subject, img_id, gender, glasses, eye_state, reflections, lighting, sensor]
         eye_state: 0 = closed, 1 = open
      2. Folder naming conventions ('open', 'closed', 'awake', 'sleepy').
    """
    # 1. Check MRL standard naming convention (underscore separated)
    parts = file_path.stem.split("_")
    if len(parts) >= 5 and parts[4] in ("0", "1"):
        return "open" if parts[4] == "1" else "closed"

    # 2. Check parent directory names
    path_str = str(file_path).lower()
    path_parts = [p.lower() for p in file_path.parts]

    # Check for closed indications first
    if any(k in path_parts for k in ["closed", "closed_eyes", "closed_eye", "sleepy", "close"]):
        return "closed"
    if any(k in path_parts for k in ["open", "open_eyes", "open_eye", "awake"]):
        return "open"

    # Fallback to string matching in file name
    name_lower = file_path.stem.lower()
    if "closed" in name_lower or "sleepy" in name_lower:
        return "closed"
    if "open" in name_lower or "awake" in name_lower:
        return "open"

    return None


def classify_yawn_file(file_path: Path) -> str | None:
    """
    Determine whether a face/mouth image is 'yawn' or 'no_yawn'.
    Supports folder and file names like 'yawn', 'no_yawn', 'no-yawn', 'normal'.
    """
    path_parts = [p.lower() for p in file_path.parts]
    path_str = str(file_path).lower()

    # Check 'no_yawn' / normal first (negative condition)
    for p in path_parts:
        if p in ["no_yawn", "noyawn", "no-yawn", "no yawn", "normal", "non_yawn", "non-yawn"]:
            return "no_yawn"

    if "no_yawn" in path_str or "noyawn" in path_str or "no-yawn" in path_str:
        return "no_yawn"

    # Check 'yawn'
    for p in path_parts:
        if p in ["yawn", "yawning", "yawns"]:
            return "yawn"

    if "yawn" in file_path.stem.lower():
        return "yawn"

    return None


def process_mrl_dataset(search_dir: Path):
    """Scan search_dir for MRL Eye images and organize them into data/eyes/."""
    print(f"[*] Scanning for MRL Eye dataset images in '{search_dir}'...")
    if not search_dir.exists():
        print(f"[!] Source folder '{search_dir}' does not exist. Skipping.")
        return 0, 0

    open_count = 0
    closed_count = 0
    skipped_count = 0

    for root, _, files in os.walk(search_dir):
        # Skip the project's own processed directories to avoid circular copies
        root_path = Path(root).resolve()
        if EYES_OPEN_DIR.resolve() in root_path.parents or root_path == EYES_OPEN_DIR.resolve():
            continue
        if EYES_CLOSED_DIR.resolve() in root_path.parents or root_path == EYES_CLOSED_DIR.resolve():
            continue
        if YAWN_DIR.resolve() in root_path.parents or root_path == YAWN_DIR.resolve():
            continue
        if NO_YAWN_DIR.resolve() in root_path.parents or root_path == NO_YAWN_DIR.resolve():
            continue

        for file_name in files:
            file_path = root_path / file_name
            if not is_image_file(file_path):
                continue

            state = classify_mrl_file(file_path)
            if state == "open":
                safe_copy_image(file_path, EYES_OPEN_DIR, prefix=file_path.parent.name)
                open_count += 1
            elif state == "closed":
                safe_copy_image(file_path, EYES_CLOSED_DIR, prefix=file_path.parent.name)
                closed_count += 1
            else:
                skipped_count += 1

    print(f"[+] MRL Processing complete: +{open_count} open, +{closed_count} closed (unmatched: {skipped_count})")
    return open_count, closed_count


def process_yawn_dataset(search_dir: Path):
    """Scan search_dir for Yawn images and organize them into data/yawn/."""
    print(f"[*] Scanning for Yawn dataset images in '{search_dir}'...")
    if not search_dir.exists():
        print(f"[!] Source folder '{search_dir}' does not exist. Skipping.")
        return 0, 0

    yawn_count = 0
    no_yawn_count = 0
    skipped_count = 0

    for root, _, files in os.walk(search_dir):
        root_path = Path(root).resolve()
        if EYES_OPEN_DIR.resolve() in root_path.parents or root_path == EYES_OPEN_DIR.resolve():
            continue
        if EYES_CLOSED_DIR.resolve() in root_path.parents or root_path == EYES_CLOSED_DIR.resolve():
            continue
        if YAWN_DIR.resolve() in root_path.parents or root_path == YAWN_DIR.resolve():
            continue
        if NO_YAWN_DIR.resolve() in root_path.parents or root_path == NO_YAWN_DIR.resolve():
            continue

        for file_name in files:
            file_path = root_path / file_name
            if not is_image_file(file_path):
                continue

            state = classify_yawn_file(file_path)
            if state == "yawn":
                safe_copy_image(file_path, YAWN_DIR, prefix=file_path.parent.name)
                yawn_count += 1
            elif state == "no_yawn":
                safe_copy_image(file_path, NO_YAWN_DIR, prefix=file_path.parent.name)
                no_yawn_count += 1
            else:
                skipped_count += 1

    print(f"[+] Yawn Processing complete: +{yawn_count} yawn, +{no_yawn_count} no_yawn (unmatched: {skipped_count})")
    return yawn_count, no_yawn_count


def auto_detect_and_extract_archives(downloads_dir: Path):
    """Automatically extract any zip archives found in the downloads directory."""
    if not downloads_dir.exists():
        return

    for item in downloads_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".zip":
            extract_target = downloads_dir / f"extracted_{item.stem}"
            extract_zip_if_needed(item, extract_target)


def count_images_in_dir(directory: Path) -> int:
    """Count number of valid image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if is_image_file(f))


def print_summary_report():
    """Print formatted summary table showing counts in each dataset folder."""
    open_eyes = count_images_in_dir(EYES_OPEN_DIR)
    closed_eyes = count_images_in_dir(EYES_CLOSED_DIR)
    yawns = count_images_in_dir(YAWN_DIR)
    no_yawns = count_images_in_dir(NO_YAWN_DIR)

    total_eyes = open_eyes + closed_eyes
    total_yawn = yawns + no_yawns

    print("\n" + "=" * 62)
    print("           DATASET ORGANIZATION SUMMARY REPORT")
    print("=" * 62)
    print(f"{'Category':<12} | {'Class':<14} | {'Image Count':<12} | {'Status':<10}")
    print("-" * 62)
    print(f"{'Eyes':<12} | {'open':<14} | {open_eyes:<12} | {'READY' if open_eyes > 0 else 'EMPTY'}")
    print(f"{'Eyes':<12} | {'closed':<14} | {closed_eyes:<12} | {'READY' if closed_eyes > 0 else 'EMPTY'}")
    print("-" * 62)
    print(f"Total Eye Dataset Images : {total_eyes:,}")
    print("-" * 62)
    print(f"{'Yawn':<12} | {'yawn':<14} | {yawns:<12} | {'READY' if yawns > 0 else 'EMPTY'}")
    print(f"{'Yawn':<12} | {'no_yawn':<14} | {no_yawns:<12} | {'READY' if no_yawns > 0 else 'EMPTY'}")
    print("-" * 62)
    print(f"Total Yawn Dataset Images: {total_yawn:,}")
    print("=" * 62)

    # Sanity checks and suggestions
    all_ready = (open_eyes > 0 and closed_eyes > 0 and yawns > 0 and no_yawns > 0)
    if all_ready:
        print("[SUCCESS] All 4 dataset classes are populated and ready for training!")
    else:
        print("\n[!] NOTICE: Some folders are currently empty.")
        print("    If you just downloaded the datasets, make sure to specify the raw data path:")
        print("    Example: python src/prepare_data.py --source-dir data/downloads")
    print("=" * 62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare and organize MRL Eye and Yawn datasets.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/downloads",
        help="Base folder where raw datasets or zips are stored (default: data/downloads)."
    )
    parser.add_argument(
        "--mrl-dir",
        type=str,
        default=None,
        help="Specific path to raw MRL Eye dataset."
    )
    parser.add_argument(
        "--yawn-dir",
        type=str,
        default=None,
        help="Specific path to raw YawnDD / Yawn dataset."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip organization and only print current dataset summary count."
    )

    args = parser.parse_args()

    ensure_target_directories()

    if args.report_only:
        print_summary_report()
        return

    source_base = Path(args.source_dir).resolve()
    print(f"[*] Base source directory: {source_base}")

    # Auto-extract any zip files in downloads
    if source_base.exists():
        auto_detect_and_extract_archives(source_base)

    # Resolve MRL directory
    mrl_source = Path(args.mrl_dir).resolve() if args.mrl_dir else source_base
    process_mrl_dataset(mrl_source)

    # Resolve Yawn directory
    yawn_source = Path(args.yawn_dir).resolve() if args.yawn_dir else source_base
    process_yawn_dataset(yawn_source)

    # Print final summary
    print_summary_report()


if __name__ == "__main__":
    main()
