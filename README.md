# Rock Paper Scissors Gesture Recognition

A real-time hand gesture recognition system for Rock-Paper-Scissors using MediaPipe Hands and machine learning.

## Features

- **Landmark-based detection**: Uses MediaPipe's 21 hand landmarks for robust gesture recognition
- **Lightweight classifier**: LogisticRegression model trained on normalized hand features
- **Real-time inference**: Live webcam gesture recognition with confidence scores
- **Easy data collection**: Interactive script to collect training samples
- **Cross-platform**: Works on any system with a webcam

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

## How It Works

1. **Feature Extraction**: Extracts 21 hand landmarks using MediaPipe
2. **Normalization**: Translates to wrist position and scales by hand size
3. **Classification**: Uses LogisticRegression on 63-dimensional feature vector
4. **Real-time**: Processes webcam feed at ~30 FPS

## Tips for Best Results

- Vary distance and angles during data collection
- Ensure good lighting conditions
- Collect samples with both hands
- Keep gestures steady during capture
- Aim for balanced dataset (equal samples per class)
