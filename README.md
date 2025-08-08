# 🎬 Video Stabilization & Comparison Toolkit (Python Version)

> **This project is a Python implementation of [Lakshya-Kejriwal/Real-Time-Video-Stabilization](https://github.com/Lakshya-Kejriwal/Real-Time-Video-Stabilization).**  
> The original algorithm is described in the paper: [A Hybrid Filtering Approach of Digital Video Stabilization for UAV Using Kalman and Low Pass Filter](https://www.sciencedirect.com/science/article/pii/S1877050916314624).

---

## 🚀 Project Overview

A powerful, easy-to-use toolkit for video stabilization and multi-algorithm comparison, based on Python and OpenCV. Supports single video stabilization, batch comparison, interactive GUI comparison, and more. Perfect for de-shaking, algorithm evaluation, and academic research.

---

## 📁 Directory Structure

```
video-stable-v1/
│
├── main.py                        # Main entry, single video stabilization example
├── video_stabilizer.py            # Video stabilization algorithm (Kalman filter + affine transform)
│
├── compare/
│   ├── batch_comparison.py        # Batch video comparison tool (CLI)
│   ├── interactive_comparison.py  # Interactive comparison tool (Tkinter GUI)
│   └── video_stabilization_comparison.py # Core class for multi-video comparison
│
├── comparison_output.mp4          # Example comparison output
├── stabilized_output_kalman.avi   # Example stabilized output
├── stabilized_output.avi          # Other output example
└── ...
```

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- numpy
- tkinter (standard library, required for GUI)

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## ✨ Features

- **Real-time video stabilization** using Kalman filter and affine transformation
- **Batch video comparison** for algorithm evaluation
- **Interactive GUI** for visual comparison and result export
- **Easy integration** and extensibility for research and development
- **Feature point detection algorithm selectable:** Supports 'gftt' (default), 'orb', 'sift', and can choose different feature point extraction methods based on requirements.

---

## 📦 Module Descriptions

### `main.py`
- Single video stabilization demo
- Supports webcam or local video file input
- Outputs to `stabilized_output_kalman.avi`
- **Feature type selectable:** Set `feature_type` to `'gftt'` (default), `'orb'`, or `'sift'` in the code to choose the feature detection algorithm.

### `video_stabilizer.py`
- Core stabilization algorithm (feature tracking + affine + Kalman filter)
- Automatic cropping and resolution recovery

### `compare/batch_comparison.py`
- Command-line batch comparison tool
- Compares all videos in a folder and generates a composite video

### `compare/interactive_comparison.py`
- Tkinter-based interactive GUI
- Drag-and-drop, order adjustment, real-time preview, export

### `compare/video_stabilization_comparison.py`
- Core class for multi-video comparison
- Frame sync, label display, grid stitching, output saving

---

## 🚦 Usage

### 1. Single Video Stabilization
Edit `main.py` to set your input video path and select the feature detection algorithm:

```python
feature_type = "gftt"  # Options: "gftt" (default), "orb", "sift"
stabilizer = VideoStabilizer(feature_type=feature_type)
```

Then run:
```bash
python main.py
```
Output: `stabilized_output_kalman.avi`

### 2. Batch Video Comparison
```bash
python compare/batch_comparison.py
```
Follow the prompt to enter the video folder path. A comparison video will be generated automatically.

### 3. Interactive Comparison (Recommended)
```bash
python compare/interactive_comparison.py
```
A GUI will pop up. Add multiple videos, set the output path, and click "Start Comparison".

---

## ❓ FAQ

- **Black border issue**: The algorithm does not automatically remove all black edges. Some cropping is performed, but black borders may still appear depending on the transformation.
- **Resolution/frame rate mismatch**: The comparison tool adapts automatically, but it is recommended that input videos have the same parameters.
- **Missing dependencies**: Please ensure all required libraries are installed.

---

## 🤝 Acknowledgements

- **Original C++ Project:** [Lakshya-Kejriwal/Real-Time-Video-Stabilization](https://github.com/Lakshya-Kejriwal/Real-Time-Video-Stabilization)
- **Original Paper:** [Real Time Video Stabilization Using Kalman Filter](https://www.sciencedirect.com/science/article/pii/S1877050916314624)

---

## 📬 Contact & Contribution

Feel free to submit Issues or Pull Requests for suggestions, bug reports, or contributions!

---


*If you need a more detailed technical description or a Chinese version, please let me know!*
