"""Configuration and constants for Smart Road Scene Analyzer."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# Model presets
IMAGE_DEFAULT_MODEL = "yolov8m.pt"  # Automatically highest accuracy for images
VIDEO_MODELS: Dict[str, str] = {
    "YOLOv8n (Fast)": "yolov8n.pt",
    "YOLOv8m (High Accuracy)": "yolov8m.pt",
}

# Inference Resolution (1280px High-Res standard for deep road detection)
DEFAULT_INFERENCE_SIZE = 1280

# Backend Video Processing Defaults (Fast CCTV default without UI slider)
DEFAULT_FRAME_STRIDE = 2

# Temporal Tracking Persistence (Video)
DEFAULT_MIN_TRACK_PERSISTENCE = 3  # Object must persist in at least 3 frames

# Road Region of Interest (ROI) Defaults
DEFAULT_ROI_TOP_PCT = 0.25  # Exclude top 25% (sky/horizon)
DEFAULT_ROI_SIDE_PCT = 0.0  # Exclude side margins (sidewalks/poles)

# COCO Class IDs relevant for road scenes
ROAD_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Category classification
VEHICLE_CLASSES: List[str] = ["car", "motorcycle", "bus", "truck", "bicycle"]
PEDESTRIAN_CLASSES: List[str] = ["person"]

# BGR colors for clear, professional bounding boxes
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (245, 130, 48),     # Coral/Orange
    "bicycle": (60, 180, 75),      # Green
    "car": (230, 25, 75),          # Red/Crimson
    "motorcycle": (0, 130, 200),   # Blue
    "bus": (145, 30, 180),         # Purple
    "truck": (240, 50, 230),       # Magenta
}
DEFAULT_COLOR_BGR = (128, 128, 128)

# Detection & NMS Threshold Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_IOU_THRESHOLD = 0.50

# Class-specific confidence thresholds to prevent false positives while preserving real targets
CLASS_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "car": 0.50,         # Cars are distinct; 0.50 stops manholes and road marks
    "person": 0.45,      # Pedestrians have varied pose; 0.45 catches real persons
    "motorcycle": 0.45,  # Two-wheelers
    "bicycle": 0.45,
    "bus": 0.50,
    "truck": 0.50,
}

# Bounding box size & aspect-ratio filters (to reject poles, road markings, manholes, speckles)
MIN_BBOX_WIDTH = 20      # minimum width in pixels
MIN_BBOX_HEIGHT = 20     # minimum height in pixels
MIN_BBOX_AREA = 400      # minimum area in pixels^2 (20x20)

CLASS_GEOMETRY_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "person": {
        "min_width": 14,
        "min_height": 22,
        "min_aspect_ratio": 0.15,  # w / h (rejects ultra-thin vertical poles)
        "max_aspect_ratio": 1.30,  # Real pedestrians are vertical, not flat stripes
    },
    "car": {
        "min_width": 24,
        "min_height": 18,
        "min_aspect_ratio": 0.40,  # Rejects vertical poles and thin dividers
        "max_aspect_ratio": 4.50,  # Rejects elongated flat lane stripes
    },
    "motorcycle": {
        "min_width": 16,
        "min_height": 16,
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 3.00,
    },
    "bicycle": {
        "min_width": 16,
        "min_height": 16,
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 3.00,
    },
    "bus": {
        "min_width": 28,
        "min_height": 24,
        "min_aspect_ratio": 0.45,
        "max_aspect_ratio": 5.00,
    },
    "truck": {
        "min_width": 28,
        "min_height": 24,
        "min_aspect_ratio": 0.40,
        "max_aspect_ratio": 5.00,
    },
}

# Traffic Density Thresholds (based on total vehicle count in frame/scene)
TRAFFIC_DENSITY_THRESHOLDS = {
    "low_max": 3,      # 0 to 3: Low
    "medium_max": 9,   # 4 to 9: Medium
                       # 10+: High
}

TRAFFIC_DENSITY_LABELS = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
}

TRAFFIC_DENSITY_COLORS = {
    "Low": "#16A34A",       # Emerald green
    "Medium": "#D97706",    # Amber
    "High": "#DC2626",      # Crimson red
}
