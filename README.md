# 🛡️ Anti Face-Spoofing

**Anti Face-Spoofing** is a real-time face recognition and liveness detection system using YOLO, RetinaFace, ByteTrack, and face alignment. It captures webcam input, detects faces, checks for liveness (real vs. spoof), and tracks individuals using assigned IDs.

---

## 🚀 Features

- 🔍 Real-time face detection using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
- 🎯 Accurate face alignment before liveness detection
- 🧠 Liveness detection using a custom YOLOv8 model
- 👁️ Multi-face tracking using [ByteTrack](https://github.com/ifzhang/ByteTrack)
- ⚡ Live FPS monitoring and display

---

## 📦 Requirements

- Python 3.8 or higher
- GPU (recommended for real-time performance)

---

## 📥 Installation

### 1. Clone this repository

```bash
git clone https://github.com/Hoanle123/face_liveness.git
cd face-liveness
```

### 2. Create a virtual environment (recommended):

 ```bash
 python3 -m venv face
 source face/bin/activate  # On Linux/macOS
 face\Scripts\activate    # On Windows
 ```
### 3. Install dependencies

 ```bash
pip install -r requirements.txt
 ```

## 🔧 Environment Configuration

The `.env` file in the project root directory has the following content:

```env
RATIO_THRESHOLD_FACE_SIZE=0.3
RATIO_THRESHOLD_STRAIGHT=0.2
RATIO_TRACKING_BBOX=0.2
LIVENESS_PATH="weights/yolo_liveness.pt"
FACEMODEL_PATH="weights/mobilenet0.25_Final.pth"
```
You can change the ratio values or test with your own model. Have fun! 😄

## ▶️ How to Run

Run the main script:
 ```bash
python main.py
 ```
### Controls

- Press **`q`** to quit the camera window anytime.

### On Screen Indicators

- 🟢 **Live** — Indicates a **real face** detected.
- 🚫 **Spoof** — Indicates a **fake face** (e.g., printed photo or video replay).


