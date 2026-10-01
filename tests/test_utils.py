"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (
    draw_detections,
    export_detections_to_dataframe,
    export_summary_to_csv,
    get_class_color,
)


def test_get_class_color():
    car_color = get_class_color("car")
    assert isinstance(car_color, tuple)
    assert len(car_color) == 3

    unknown_color = get_class_color("spaceship")
    assert isinstance(unknown_color, tuple)
    assert len(unknown_color) == 3


def test_draw_detections():
    # Synthetic blank image 400x400x3
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.88,
            "bbox_xyxy": [50.0, 50.0, 150.0, 150.0],
            "track_id": 1,
        },
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.92,
            "bbox_xyxy": [200.0, 100.0, 240.0, 220.0],
            "track_id": 2,
        },
    ]

    density_badge = {"level": "Low", "vehicle_count": 1}
    annotated = draw_detections(img, detections, density_badge=density_badge)

    assert annotated.shape == img.shape
    # Image should not be all black anymore since bounding boxes and tags were drawn
    assert np.any(annotated > 0)


def test_export_detections_to_dataframe():
    detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.88,
            "bbox_xyxy": [50.0, 50.0, 150.0, 150.0],
            "track_id": 4,
        }
    ]
    df = export_detections_to_dataframe(detections)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Track ID" in df.columns
    assert "Confidence" in df.columns
    assert df.iloc[0]["Class"] == "car"
    assert df.iloc[0]["Track ID"] == 4


def test_export_summary_to_csv(tmp_path: Path):
    summary_data = {
        "class_counts": {"car": 3, "person": 1},
        "category_counts": {"vehicles": 3, "pedestrians": 1, "total": 4},
        "density": {"level": "Low", "description": "Light traffic"},
    }
    out_file = tmp_path / "summary.csv"
    res_path = export_summary_to_csv(summary_data, out_file)

    assert res_path.exists()
    df = pd.read_csv(res_path)
    assert "Metric" in df.columns
    assert "Value" in df.columns
    assert "Count: car" in df["Metric"].values
