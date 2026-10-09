# Real-Time Drowsiness Detection System for Indian Truck Drivers

An AI-powered computer vision and deep learning system designed to detect driver drowsiness and fatigue in real-time under challenging cabin conditions (night driving, poor lighting, glare, vibrations).

## Project Structure
```text
Drowsiness Detection/
├── app/                  # Application UI (PySide6 dashboard / alerts)
├── data/                 # Training and calibration datasets
│   ├── eyes/             # Eye state samples (open/closed)
│   └── yawn/             # Mouth state samples (yawn/normal)
├── models/               # Pretrained weights and exported models
├── src/                  # Core detection logic, feature extractors, utilities
│   └── test_setup.py     # Environment & webcam verification script
├── venv/                 # Python 3.13 virtual environment
├── README.md
└── requirements.txt      # Pinned project dependencies
```

## Setup & Getting Started

### 1. Activate the Virtual Environment (Windows PowerShell)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Camera & Environment
```powershell
python src/test_setup.py
```
