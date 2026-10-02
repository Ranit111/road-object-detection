"""End-to-end integration tests using real YOLOv8n model and real sample media."""

from pathlib import Path
import pytest

from src.config import ROAD_CLASSES
from src.detector import RoadObjectDetector
from src.traffic_analyzer import TrafficAnalyzer
from src.video_processor import VideoProcessor

SAMPLE_IMAGE = Path("samples/sample_road.jpg")
SAMPLE_VIDEO = Path("samples/sample_traffic.mp4")


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="Sample image not present")
def test_real_image_detection_pipeline():
    detector = RoadObjectDetector()
    analyzer = TrafficAnalyzer()

    detections, result = detector.detect_image(
        SAMPLE_IMAGE,
        conf_threshold=0.50,
        iou_threshold=0.50,
    )

    assert result is not None
    # Real image bus.jpg contains at least a bus and people
    assert len(detections) > 0

    detected_classes = {d["class_name"] for d in detections}
    assert "bus" in detected_classes, f"Bus was expected in detections, got: {detected_classes}"
    assert "person" in detected_classes, f"Person was expected in detections, got: {detected_classes}"

    # Verify confidence and coordinates are valid
    for d in detections:
        assert 0.0 <= d["confidence"] <= 1.0
        assert len(d["bbox_xyxy"]) == 4
        assert d["bbox_xyxy"][2] >= d["bbox_xyxy"][0]
        assert d["bbox_xyxy"][3] >= d["bbox_xyxy"][1]

    # Verify traffic analysis
    analysis = analyzer.analyze_frame(detections)
    assert analysis["total_detections"] == len(detections)
    assert analysis["density"]["level"] in ["Low", "Medium", "High"]
    assert analysis["category_counts"]["total"] == len(detections)


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not present")
def test_real_video_tracking_pipeline(tmp_path: Path):
    detector = RoadObjectDetector()
    analyzer = TrafficAnalyzer()
    proc = VideoProcessor(detector=detector, traffic_analyzer=analyzer)

    out_video = tmp_path / "processed_output.mp4"

    # Process first 10 frames to verify tracking and writer
    summary = proc.process_video(
        input_path=SAMPLE_VIDEO,
        output_path=out_video,
        conf_threshold=0.50,
        iou_threshold=0.50,
        use_tracking=True,
        frame_stride=3,
    )

    assert summary["total_frames_analyzed"] > 0
    assert summary["avg_fps"] > 0
    assert summary["peak_density"] in ["Low", "Medium", "High"]
    assert out_video.exists()
    assert out_video.stat().st_size > 0
