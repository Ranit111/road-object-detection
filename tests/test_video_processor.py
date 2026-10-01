"""Unit tests for VideoProcessor logic and temporal persistence filtering."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.traffic_analyzer import TrafficAnalyzer
from src.video_processor import VideoProcessor


def test_video_processor_initialization():
    mock_detector = MagicMock()
    analyzer = TrafficAnalyzer()
    proc = VideoProcessor(detector=mock_detector, traffic_analyzer=analyzer)
    assert proc.detector == mock_detector
    assert proc.analyzer == analyzer


def test_video_processor_file_not_found():
    mock_detector = MagicMock()
    proc = VideoProcessor(detector=mock_detector)
    with pytest.raises(FileNotFoundError):
        proc.process_video("non_existent_file_path.mp4")


def test_temporal_persistence_filtering():
    mock_detector = MagicMock()
    analyzer = TrafficAnalyzer()
    proc = VideoProcessor(detector=mock_detector, traffic_analyzer=analyzer)

    # Simulate 3 frames returned by cap.read()
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, True, True, True, False]
    mock_cap.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None),
    ]
    mock_cap.get.side_effect = lambda prop: 3 if prop == 7 else (25.0 if prop == 5 else 100)

    # Frame 1: Car #1 and Noise #99 (shadow)
    # Frame 2: Car #1
    # Frame 3: Car #1
    mock_detector.track_frame.side_effect = [
        ([{"class_name": "car", "confidence": 0.8, "bbox_xyxy": [10, 10, 50, 50], "track_id": 1},
          {"class_name": "car", "confidence": 0.55, "bbox_xyxy": [60, 60, 80, 80], "track_id": 99}], None),
        ([{"class_name": "car", "confidence": 0.82, "bbox_xyxy": [12, 12, 52, 52], "track_id": 1}], None),
        ([{"class_name": "car", "confidence": 0.85, "bbox_xyxy": [14, 14, 54, 54], "track_id": 1}], None),
    ]

    with patch("src.video_processor.cv2.VideoCapture", return_value=mock_cap), \
         patch("pathlib.Path.is_file", return_value=True):

        summary = proc.process_video(
            input_path="dummy.mp4",
            use_tracking=True,
            min_track_persistence=3,
        )

        # Track 99 only appeared in frame 1 -> rejected!
        # Track 1 appeared in all 3 frames -> confirmed!
        assert summary["unique_tracked_total"] == 1
        assert summary["final_class_counts"] == {"car": 1}
