# Smart Road Scene Analyzer

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://road-object-detection.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Ranit111%2Froad--object--detection-blue?logo=github)](https://github.com/Ranit111/road-object-detection)

A high-performance computer vision web application for intelligent road traffic analysis, vehicle counting, and traffic density estimation using **Ultralytics YOLOv8** and **Streamlit**.

---

## 🌐 Live Application
Access the deployed application directly in your browser:
👉 **[https://road-object-detection.streamlit.app/](https://road-object-detection.streamlit.app/)**

---

## 🌟 Key Features

1. **Multi-Class Road Object Detection**:
   - Detects and categorizes road participants: **cars, buses, trucks, motorcycles, bicycles, and pedestrians**.
   - Dual theme support (**Dark Mode & Light Mode**) with clear bounding box visualization.

2. **Model Variants & Accuracy Scaling**:
   - **`yolov8n.pt` (Nano)**: Optimized for fast, low-latency execution.
   - **`yolov8m.pt` (Medium)**: Enhanced precision for dense, complex traffic intersections.

3. **Temporal Tracking Persistence Filter (Video)**:
   - Integrates multi-object tracking (ByteTrack) with configurable frame persistence.
   - Filters out temporary flickering false positives (e.g. road shadows, manhole reflections).

4. **Road Region of Interest (ROI) Masking**:
   - Excludes sky, clouds, and building roofs to focus inference specifically on asphalt and traffic lanes.
   - Configurable vertical horizon and side margin cutoffs.

5. **Traffic Density Estimation**:
   - 🟢 **Low**: 0 to 3 vehicles (smooth, free-flowing traffic)
   - 🟡 **Medium**: 4 to 9 vehicles (moderate vehicle presence)
   - 🔴 **High**: 10+ vehicles (heavy congestion)

6. **Export & Analytics**:
   - Download annotated high-resolution images (`.jpg`).
   - Download processed videos with tracking IDs and annotations (`.mp4`).
   - Export detailed detection metrics and per-frame logs (`.csv`).

---

## 📁 Project Structure

```
road-object-detection/
├── .streamlit/
│   └── config.toml            # Streamlit theme & UI styling configuration
├── models/
│   └── (YOLOv8 weights auto-downloaded & cached)
├── samples/
│   ├── sample_road.jpg        # Demo road image
│   └── sample_traffic.mp4     # Demo traffic video
├── src/
│   ├── __init__.py
│   ├── config.py              # Road classes, thresholds, ROI & scaling constants
│   ├── detector.py            # YOLO inference, geometry filtering & NMS deduplication
│   ├── traffic_analyzer.py    # Class counts & traffic density logic
│   ├── video_processor.py     # Video tracking engine & temporal persistence filter
│   └── utils.py               # Bounding box drawing, overlays & CSV export helpers
├── app.py                     # Streamlit web application
├── packages.txt               # System-level dependencies for Linux deployment
├── requirements.txt           # Python library dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/Ranit111/road-object-detection.git
cd road-object-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
