# Datasets Guide

This project supports optional offline machine learning experiments (CNN training and validation) using benchmark driver drowsiness datasets. 

> **Note:** The primary real-time drowsiness detector runs using MediaPipe Face Mesh landmarks to compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR). Downloading these offline datasets is **not required** to run the live application (`app/main.py` or `src/drowsiness_detector.py`). They are used for the optional CNN experiment (`--show-cnn`) and offline model training (`src/train_eye_model.py`).

---

## 1. Required Datasets

### A. MRL Eye Dataset (Open vs. Closed Eyes)
- **Dataset Name**: MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)
- **Kaggle Identifier**: `imadeddinedjerarda/mrl-eye-dataset`
- **Purpose**: Contains >84,000 infrared eye crop images categorized into awake (open) and sleepy (closed) eyes across 37 subjects under varied lighting, glasses, and head angles. Used for training the legacy CNN classifier (`src/train_eye_model.py` -> `models/eye_model.keras`).
- **Download Command**:
  ```bash
  kaggle datasets download -d imadeddinedjerarda/mrl-eye-dataset -p data/downloads/mrl --unzip
  ```

### B. Drowsiness Dataset (Yawn vs. No Yawn)
- **Dataset Name**: Drowsiness Dataset (Driver Yawn & Fatigue)
- **Kaggle Identifier**: `dheerajperumandla/drowsiness-dataset`
- **Purpose**: Contains face and mouth region crops categorized into `yawn` and `no_yawn` states. Used to validate mouth aspect ratio thresholds and evaluate mouth closure classification.
- **Download Command**:
  ```bash
  kaggle datasets download -d dheerajperumandla/drowsiness-dataset -p data/downloads/yawn --unzip
  ```

---

## 2. Dataset Organization

After downloading the raw datasets into `data/downloads/`, run the dataset preparation script to standardize folder structures and file names:

```bash
python src/prepare_data.py --source-dir data/downloads
```

This organizes the raw images into clean project directories:
```text
data/
├── eyes/
│   ├── open/        # Standardized open eye images
│   └── closed/      # Standardized closed eye images
└── yawn/
    ├── yawn/        # Standardized yawning mouth images
    └── no_yawn/     # Standardized non-yawning mouth images
```

To verify the processed dataset count:
```bash
python src/prepare_data.py --report-only
```
