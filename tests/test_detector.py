"""Unit tests for RoadObjectDetector, false-positive filtering, and YOLO parsing."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.config import ROAD_CLASSES
from src.detector import (
    RoadObjectDetector,
    apply_post_nms,
    compute_box_containment,
    compute_box_iou,
    is_valid_detection_geometry,
)


def test_detector_parse_results():
    detector = object.__new__(RoadObjectDetector)
    detector.target_classes = ROAD_CLASSES
    detector._target_class_ids = list(ROAD_CLASSES.keys())

    mock_result = MagicMock()
    mock_boxes = MagicMock()

    # 2 boxes: one car (id 2) and one person (id 0)
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.0, 20.0, 100.0, 150.0],
        [200.0, 50.0, 250.0, 180.0],
    ])
    mock_conf = MagicMock()
    mock_conf.cpu.return_value.numpy.return_value = np.array([0.895, 0.724])
    mock_cls = MagicMock()
    mock_cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([2, 0])

    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = mock_conf
    mock_boxes.cls = mock_cls
    mock_boxes.id = None
    mock_boxes.__len__.return_value = 2

    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 2: "car"}

    detections = detector._parse_results(mock_result)

    assert len(detections) == 2
    assert detections[0]["class_name"] == "car"
    assert detections[0]["confidence"] == 0.895
    assert detections[0]["bbox_xyxy"] == [10.0, 20.0, 100.0, 150.0]
    assert detections[0]["track_id"] is None

    assert detections[1]["class_name"] == "person"
    assert detections[1]["confidence"] == 0.724


def test_detector_parse_with_tracking():
    detector = object.__new__(RoadObjectDetector)
    detector.target_classes = ROAD_CLASSES
    detector._target_class_ids = list(ROAD_CLASSES.keys())

    mock_result = MagicMock()
    mock_boxes = MagicMock()

    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 20.0, 100.0, 150.0]])
    mock_conf = MagicMock()
    mock_conf.cpu.return_value.numpy.return_value = np.array([0.92])
    mock_cls = MagicMock()
    mock_cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([2])
    mock_id = MagicMock()
    mock_id.cpu.return_value.numpy.return_value.astype.return_value = np.array([7])

    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = mock_conf
    mock_boxes.cls = mock_cls
    mock_boxes.id = mock_id
    mock_boxes.__len__.return_value = 1

    mock_result.boxes = mock_boxes
    mock_result.names = {2: "car"}

    detections = detector._parse_results(mock_result)
    assert len(detections) == 1
    assert detections[0]["track_id"] == 7
    assert detections[0]["class_name"] == "car"


def test_filter_small_boxes_and_speckles():
    assert not is_valid_detection_geometry("car", [10, 10, 15, 15], min_width=20, min_height=20)
    assert is_valid_detection_geometry("car", [10, 10, 80, 60], min_width=20, min_height=20)


def test_filter_extreme_aspect_ratios_poles_and_lines():
    # Ultra-thin vertical street pole (width=4, height=120)
    assert not is_valid_detection_geometry("person", [10, 10, 14, 130])

    # Elongated horizontal road stripe (width=300, height=12)
    assert not is_valid_detection_geometry("car", [0, 50, 300, 62])


def test_class_specific_thresholds():
    detector = object.__new__(RoadObjectDetector)
    detector.target_classes = ROAD_CLASSES
    detector._target_class_ids = [0, 2]

    mock_result = MagicMock()
    mock_boxes = MagicMock()

    # Box 1: car with conf 0.42 (rejected by car threshold 0.50)
    # Box 2: person with conf 0.48 (kept by person threshold 0.45)
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.0, 20.0, 100.0, 100.0],
        [200.0, 50.0, 240.0, 170.0],
    ])
    mock_conf = MagicMock()
    mock_conf.cpu.return_value.numpy.return_value = np.array([0.42, 0.48])
    mock_cls = MagicMock()
    mock_cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([2, 0])

    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = mock_conf
    mock_boxes.cls = mock_cls
    mock_boxes.id = None
    mock_boxes.__len__.return_value = 2

    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 2: "car"}

    detections = detector._parse_results(mock_result, conf_threshold=0.30)
    assert len(detections) == 1
    assert detections[0]["class_name"] == "person"
    assert detections[0]["confidence"] == 0.48


def test_post_nms_deduplication():
    dets = [
        {"class_name": "car", "confidence": 0.70, "bbox_xyxy": [10.0, 10.0, 100.0, 80.0]},
        {"class_name": "car", "confidence": 0.90, "bbox_xyxy": [12.0, 11.0, 102.0, 82.0]},
    ]
    cleaned = apply_post_nms(dets, iou_threshold=0.50)
    assert len(cleaned) == 1
    assert cleaned[0]["confidence"] == 0.90


def test_road_roi_filtering():
    detector = object.__new__(RoadObjectDetector)
    detector.target_classes = ROAD_CLASSES
    detector._target_class_ids = [2]

    mock_result = MagicMock()
    mock_result.orig_shape = (1000, 1000)  # height=1000, width=1000
    mock_boxes = MagicMock()

    # Box 1: In the sky zone (y1=50, y2=150) -> y_center=100 < 250 (top 25%)
    # Box 2: On the road (y1=400, y2=600) -> y_center=500 > 250
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([
        [100.0, 50.0, 250.0, 150.0],
        [300.0, 400.0, 500.0, 600.0],
    ])
    mock_conf = MagicMock()
    mock_conf.cpu.return_value.numpy.return_value = np.array([0.85, 0.92])
    mock_cls = MagicMock()
    mock_cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([2, 2])

    mock_boxes.xyxy = mock_xyxy
    mock_boxes.conf = mock_conf
    mock_boxes.cls = mock_cls
    mock_boxes.id = None
    mock_boxes.__len__.return_value = 2

    mock_result.boxes = mock_boxes
    mock_result.names = {2: "car"}

    # Exclude top 25% (250 pixels)
    detections = detector._parse_results(mock_result, roi_top_pct=0.25)
    assert len(detections) == 1
    assert detections[0]["bbox_xyxy"] == [300.0, 400.0, 500.0, 600.0]
