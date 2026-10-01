"""Utility functions for image annotation, styling, and file handling."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src.config import CLASS_COLORS_BGR, DEFAULT_COLOR_BGR


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color assigned to an object class."""
    return CLASS_COLORS_BGR.get(class_name.lower(), DEFAULT_COLOR_BGR)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_conf: bool = True,
    show_track_id: bool = True,
    density_badge: Optional[Dict[str, Any]] = None,
    line_thickness: int = 2,
    roi_top_pct: float = 0.0,
    roi_side_pct: float = 0.0,
) -> np.ndarray:
    """
    Annotate image with bounding boxes, labels, confidence scores, and optional tracking IDs.
    Produces clean, professional, unobtrusive computer-vision graphics.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Draw Road Horizon ROI guide line if active
    roi_start_y = 0
    if 0.05 <= roi_top_pct <= 0.60:
        roi_start_y = int(h * roi_top_pct)
        cv2.line(annotated, (0, roi_start_y), (w, roi_start_y), (255, 200, 0), 1, cv2.LINE_AA)
        roi_text = f"Road Horizon Boundary (Top {int(roi_top_pct * 100)}% Excluded)"
        cv2.putText(
            annotated,
            roi_text,
            (w - 330, max(15, roi_start_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

    # Draw Vertical Road Boundary lines if active
    if 0.02 <= roi_side_pct <= 0.40:
        left_x = int(w * roi_side_pct)
        right_x = int(w * (1.0 - roi_side_pct))
        cv2.line(annotated, (left_x, roi_start_y), (left_x, h), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (right_x, roi_start_y), (right_x, h), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"Road Lane Boundary (Sides {int(roi_side_pct * 100)}% Excluded)",
            (left_x + 8, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    for det in detections:
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cls_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id")

        color = get_class_color(cls_name)

        # Draw crisp bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        if show_labels:
            parts = []
            if show_track_id and track_id is not None:
                parts.append(f"#{track_id}")
            parts.append(cls_name)
            if show_conf:
                parts.append(f"{conf:.2f}")
            label = " ".join(parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

            bg_y1 = max(0, y1 - text_h - baseline - 4)
            bg_y2 = y1
            bg_x1 = x1
            bg_x2 = min(w, x1 + text_w + 6)

            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            text_pos = (bg_x1 + 3, bg_y2 - baseline - 1)
            cv2.putText(annotated, label, text_pos, font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    # Optional density badge overlay in top-left
    if density_badge:
        level = density_badge.get("level", "Low")
        count = density_badge.get("vehicle_count", 0)
        badge_text = f"Traffic: {level} ({count} vehicles)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thick = 2
        (bw, bh), bl = cv2.getTextSize(badge_text, font, font_scale, font_thick)

        cv2.rectangle(annotated, (15, 15), (25 + bw, 35 + bh), (30, 30, 30), -1)
        border_color = (40, 180, 40) if level == "Low" else ((0, 165, 255) if level == "Medium" else (0, 0, 220))
        cv2.rectangle(annotated, (15, 15), (25 + bw, 35 + bh), border_color, 2)
        cv2.putText(annotated, badge_text, (20, 25 + bh), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return annotated


def save_annotated_image(image_bgr: np.ndarray, output_path: Union[str, Path]) -> Path:
    """Save an annotated BGR image to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image_bgr)
    return out


def export_detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert detections list to a clean Pandas DataFrame."""
    rows = []
    for d in detections:
        bbox = d.get("bbox_xyxy", [0, 0, 0, 0])
        rows.append({
            "Track ID": d.get("track_id", "-"),
            "Class": d.get("class_name", ""),
            "Confidence": d.get("confidence", 0.0),
            "X1": int(round(bbox[0])) if len(bbox) == 4 else 0,
            "Y1": int(round(bbox[1])) if len(bbox) == 4 else 0,
            "X2": int(round(bbox[2])) if len(bbox) == 4 else 0,
            "Y2": int(round(bbox[3])) if len(bbox) == 4 else 0,
        })
    return pd.DataFrame(rows)


def export_summary_to_csv(summary_data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Export detection metrics & counts to CSV format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls_name, count in summary_data.get("class_counts", {}).items():
        rows.append({"Metric": f"Count: {cls_name}", "Value": count})

    cat_counts = summary_data.get("category_counts", {})
    rows.append({"Metric": "Total Vehicles", "Value": cat_counts.get("vehicles", 0)})
    rows.append({"Metric": "Total Pedestrians", "Value": cat_counts.get("pedestrians", 0)})
    rows.append({"Metric": "Total Objects", "Value": cat_counts.get("total", 0)})

    density = summary_data.get("density", {})
    rows.append({"Metric": "Traffic Density Level", "Value": density.get("level", "N/A")})
    rows.append({"Metric": "Traffic Density Description", "Value": density.get("description", "N/A")})

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out
