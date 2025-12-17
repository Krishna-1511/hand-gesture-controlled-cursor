# Hand Gesture Controlled Cursor

A real-time hand gesture based system to control mouse and system actions using a webcam.

## Features
- Cursor movement
- Left and right click
- Scroll
- Drag and drop
- Zoom in / Zoom out
- Volume control
- Window switching
- Play / Stop actions

## Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## How It Works
- Uses MediaPipe to detect hand landmarks from webcam input.
- Gestures are identified based on finger positions and landmark distances.
- Each gesture is mapped to a specific system action.
- Smoothing and cooldown mechanisms are applied to reduce noise and prevent false triggers.

## Setup & Run
```bash
pip install -r requirements.txt
python main.py

