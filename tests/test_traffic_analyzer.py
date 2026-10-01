"""Unit tests for TrafficAnalyzer."""

import pytest
from src.traffic_analyzer import TrafficAnalyzer


def test_count_by_class():
    analyzer = TrafficAnalyzer()
    detections = [
        {"class_name": "car", "confidence": 0.88},
        {"class_name": "car", "confidence": 0.72},
        {"class_name": "bus", "confidence": 0.91},
        {"class_name": "person", "confidence": 0.65},
    ]
    counts = analyzer.count_by_class(detections)
    assert counts == {"car": 2, "bus": 1, "person": 1}


def test_get_category_counts():
    analyzer = TrafficAnalyzer()
    detections = [
        {"class_name": "car"},
        {"class_name": "motorcycle"},
        {"class_name": "truck"},
        {"class_name": "person"},
        {"class_name": "person"},
    ]
    cat_counts = analyzer.get_category_counts(detections)
    assert cat_counts["vehicles"] == 3
    assert cat_counts["pedestrians"] == 2
    assert cat_counts["total"] == 5


def test_estimate_density_levels():
    analyzer = TrafficAnalyzer(low_max=3, medium_max=9)

    # Low density (0 to 3 vehicles)
    d_low = analyzer.estimate_density(2)
    assert d_low["level"] == "Low"
    assert d_low["vehicle_count"] == 2

    # Boundary Low
    d_low_bound = analyzer.estimate_density(3)
    assert d_low_bound["level"] == "Low"

    # Medium density (4 to 9 vehicles)
    d_med = analyzer.estimate_density(6)
    assert d_med["level"] == "Medium"

    # Boundary Medium
    d_med_bound = analyzer.estimate_density(9)
    assert d_med_bound["level"] == "Medium"

    # High density (10+ vehicles)
    d_high = analyzer.estimate_density(10)
    assert d_high["level"] == "High"

    d_high_extreme = analyzer.estimate_density(25)
    assert d_high_extreme["level"] == "High"


def test_analyze_frame():
    analyzer = TrafficAnalyzer()
    detections = [
        {"class_name": "car", "confidence": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
        {"class_name": "bus", "confidence": 0.85, "bbox_xyxy": [60, 60, 120, 120]},
        {"class_name": "person", "confidence": 0.7, "bbox_xyxy": [5, 5, 20, 40]},
    ]
    result = analyzer.analyze_frame(detections)
    assert result["total_detections"] == 3
    assert result["category_counts"]["vehicles"] == 2
    assert result["category_counts"]["pedestrians"] == 1
    assert result["density"]["level"] == "Low"
    assert "car" in result["class_counts"]


def test_unique_tracked_counting():
    tracked_set = {
        (1, "car"),
        (2, "car"),
        (3, "bus"),
        (4, "person"),
        (5, "bicycle"),
    }
    summary = TrafficAnalyzer.count_unique_tracked(tracked_set)
    assert summary["unique_total"] == 5
    assert summary["unique_vehicles"] == 4  # car (2) + bus (1) + bicycle (1)
    assert summary["unique_pedestrians"] == 1  # person (1)
    assert summary["unique_class_counts"]["car"] == 2
