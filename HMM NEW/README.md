# HMM Activity Recognition from Video

A Python project using Hidden Markov Models (HMMs) and object detection to classify activities like walking, loitering, and running from video data.

## Features
- Real-time person detection using MobileNet SSD
- Motion feature extraction from bounding boxes
- HMM training and state prediction
- Live webcam inference

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### 1. Train the HMM model:
```bash
python train.py path/to/video.mp4
```

### 2. Run Live Prediction:
```bash
python live_predict.py
```

## Folder Structure
```
hmm_activity_recognition/
├── train.py                # HMM training script
├── live_predict.py         # Live webcam prediction
├── object_detection_features.py  # Object detection + feature extractor
├── models/                 # Pre-trained object detection models
└── saved_models/           # Trained HMM models
```

## Dependencies
- OpenCV
- NumPy
- scikit-learn
- hmmlearn
