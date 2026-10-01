# Smart Road Scene Analyzer (Advanced Edition)

A high-accuracy, practical computer-vision application for road traffic analysis. It integrates **Ultralytics YOLO (Nano, Small, Medium, or Custom Weights)**, **High-Resolution Inference (1280px)**, **Temporal Tracking Persistence Filtering**, **Road Region of Interest (ROI) Masking**, vehicle and pedestrian counting, traffic density classification, and comprehensive CSV/video downloads.

---

## 🌟 Advanced Features

1. **Model Architecture Scaling**:
   - **`yolov8n.pt` (Nano)**: Ultra-fast, ideal for low-power CPUs.
   - **`yolov8s.pt` (Small)**: Recommended default — delivers **+20% to +30% mAP accuracy boost** while maintaining high FPS.
   - **`yolov8m.pt` (Medium)**: High precision for dense, complex traffic intersections.
   - **Custom Weights (`.pt`)**: Support for fine-tuned weights (e.g. models trained on Kaggle/Roboflow road traffic datasets with rickshaws, auto-rickshaws, vans, etc.).

2. **High-Resolution Inference (`imgsz=1280`)**:
   - Easily toggle between `Standard (640px)` and `High-Res (1280px)` to catch small, distant bicycles, pedestrians, or vehicles far down the road without missing detections.

3. **Temporal Tracking Persistence Filter (Video)**:
   - Configurable persistence filter (default: `3 frames`).
   - A detected object is only confirmed and displayed after its track ID persists across at least N frames.
   - **Instantly eliminates 1–2 frame flickering false positives** caused by road asphalt shadows, manhole reflections, or sunlight glare.

4. **Road Region of Interest (ROI) Masking**:
   - Road scene photos often contain 25–40% sky, clouds, tree tops, and tall building roofs.
   - Toggle Road ROI to exclude the top horizon zone (configurable 10%–50% cutoff), focusing inference strictly on the road asphalt.

5. **False-Positive Elimination Engine**:
   - Class-specific thresholds (`car`: 0.50, `person`: 0.45, `motorcycle`: 0.45, etc.).
   - Minimum bounding box size filtering (`width >= 20px`, `height >= 20px`, `area >= 400px²`).
   - Aspect ratio geometry constraints to reject vertical street poles (`aspect < 0.15`) and horizontal lane markings (`aspect > 4.5`).
   - Secondary Post-NMS and containment suppression to remove duplicate or nested boxes.

6. **Traffic Density Estimation**:
   - 🟢 **Low**: 0 to 3 vehicles (smooth, free-flowing traffic)
   - 🟡 **Medium**: 4 to 9 vehicles (moderate vehicle presence)
   - 🔴 **High**: 10+ vehicles (heavy congestion)

7. **Export & Download Pipeline**:
   - Download annotated high-resolution images (`.jpg`).
   - Download processed videos (`.mp4`).
   - Download summary metrics and per-frame tracking time-series logs (`.csv`).

---

## 📁 Project Architecture

```
ROAD OBJECT DETECTION/
│
├── ARCHITECTURE/               # Product, technical, design & agent specs
│   ├── agent_instruction.md
│   ├── design.md
│   ├── prd.md
│   └── trd.md
│
├── models/
│   ├── yolov8n.pt             # Cached YOLOv8n weights
│   └── (yolov8s.pt / yolov8m.pt auto-downloaded on selection)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Class mappings, thresholds, ROI & scaling constants
│   ├── detector.py            # YOLO inference, dynamic models, ROI & ByteTrack wrapper
│   ├── traffic_analyzer.py    # Class counts & traffic density logic
│   ├── video_processor.py     # Temporal persistence filter, video engine & FPS counter
│   └── utils.py               # Bounding box drawing, ROI boundary overlay, CSV/image helpers
│
├── samples/
│   ├── prepare_samples.py     # Script to generate/download test media
│   ├── sample_road.jpg        # Demo road image
│   └── sample_traffic.mp4     # Demo traffic video
│
├── tests/
│   ├── test_detector.py       # Detection, ROI, size, and geometry unit tests
│   ├── test_traffic_analyzer.py # Counting & traffic density tests
│   ├── test_utils.py          # Annotation, coloring & CSV export tests
│   ├── test_video_processor.py # Video engine & temporal persistence tests
│   └── test_integration_e2e.py # Real YOLO end-to-end integration tests
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation and run instructions
```

---

## 🛠️ Quick Start

### 1. Launch the Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run All Automated Tests
```bash
pytest tests -v
```
*(All 21 unit and integration tests will execute and pass 100%.)*
