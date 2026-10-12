# 01 · Virtual Painter

Draw in the air with your finger. A real-time, gesture-controlled drawing app built with
**OpenCV**, **MediaPipe** and **FastAPI**: the webcam tracks your hand, gestures switch
between drawing, erasing and picking colors, and a web dashboard streams the result.

> Originally developed as [Painter-OpenCV](https://github.com/yandouziyassine/Painter-OpenCV).

<!-- TODO: record a 5–10 s demo and add it here: ![demo](docs/demo.gif) -->

## Gestures

| Gesture | Action |
|---|---|
| ✊ Fist (0 fingers) | **Idle**: move freely, no drawing |
| ☝️ Index finger up | **Draw** with the fingertip |
| ✌️ Two fingers up | **Erase** with a circular eraser at the midpoint |
| ✌️ over a color swatch | **Select** that color |

Keyboard (desktop mode): `c` clear canvas · `e` toggle eraser · `q` / `Esc` quit.

## Features

- **Real-time hand tracking**: 21 MediaPipe hand landmarks per frame, smoothed with a rolling average.
- **Gesture engine**: finger-count classification with debouncing so modes don't flicker.
- **Virtual toolbar**: color swatches and buttons you "touch" in the video feed.
- **Transparent canvas layer**: a NumPy BGRA overlay blended onto the camera frame.
- **Web dashboard**: live MJPEG stream, FPS/mode/color status, camera on/off, clean shutdown.
- **Threaded pipeline**: the CV loop runs in its own thread so the web server stays responsive.

## Computer vision concepts covered

Hand landmark detection · gesture classification from landmark geometry · temporal smoothing
and debouncing · alpha compositing · real-time video I/O · streaming frames over HTTP.

## Project structure

```
01-virtual-painter/
├── main.py              # desktop app (OpenCV window)
├── Server.py            # web app (FastAPI + MJPEG stream)
├── hand_tracker.py      # MediaPipe wrapper
├── gesture_detector.py  # finger counting → IDLE / DRAW / ERASE
├── canvas.py            # transparent BGRA drawing layer
├── toolbar.py           # on-screen color swatches and buttons
├── config.py            # all tunable constants (camera, stroke size, palette…)
└── static/index.html    # dashboard UI
```

## Run it

From the repo root, on Windows:

```bash
cd 01-virtual-painter
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then either:

```bash
python main.py      # desktop window
python Server.py    # web dashboard → http://localhost:8000
```

> **Note:** `numpy` is pinned below 2 because `mediapipe 0.10.14` and older OpenCV wheels
> are compiled against NumPy 1.x. With NumPy 2, `import cv2` fails with
> `AttributeError: _ARRAY_API not found`.

## Tech stack

Python 3.10 · OpenCV · MediaPipe · NumPy · FastAPI · Uvicorn · HTML/CSS/vanilla JS
