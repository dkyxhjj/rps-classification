# Rock Paper Scissors Gesture Recognition

A real-time hand gesture recognition system for Rock-Paper-Scissors using MediaPipe Hands and machine learning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create directories:
```bash
mkdir data models
```

## Usage

### 1. Collect Training Data
```bash
python collect_data.py
```
- Press `r` for rock, `p` for paper, `s` for scissors
- Hold gesture steady while capturing
- Aim for 200-300 samples per gesture
- Press `q` to quit

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run Live Recognition
```bash
python live_inference.py
```
- Show gestures to your webcam
- Press `q` to quit

- Vary distance and angles during data collection
- Ensure good lighting conditions
- Collect samples with both hands
- Keep gestures steady during capture
- Aim for balanced dataset (equal samples per class)
