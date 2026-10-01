"""Traffic Analysis module for counting objects and estimating road traffic density."""

from typing import Any, Dict, List, Optional, Set, Tuple
from src.config import (
    PEDESTRIAN_CLASSES,
    TRAFFIC_DENSITY_COLORS,
    TRAFFIC_DENSITY_LABELS,
    TRAFFIC_DENSITY_THRESHOLDS,
    VEHICLE_CLASSES,
)


class TrafficAnalyzer:
    """Analyzes detection results to compute class counts, categories, and traffic density."""

    def __init__(
        self,
        low_max: int = TRAFFIC_DENSITY_THRESHOLDS["low_max"],
        medium_max: int = TRAFFIC_DENSITY_THRESHOLDS["medium_max"],
    ) -> None:
        self.low_max = low_max
        self.medium_max = medium_max

    def count_by_class(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count occurrences of each detected object class."""
        counts: Dict[str, int] = {}
        for det in detections:
            cls_name = det.get("class_name", "unknown")
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts

    def get_category_counts(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group detections into Vehicles, Pedestrians, and Total."""
        vehicle_count = 0
        pedestrian_count = 0

        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name in VEHICLE_CLASSES:
                vehicle_count += 1
            elif cls_name in PEDESTRIAN_CLASSES:
                pedestrian_count += 1

        return {
            "vehicles": vehicle_count,
            "pedestrians": pedestrian_count,
            "total": len(detections),
        }

    def estimate_density(self, vehicle_count: int) -> Dict[str, Any]:
        """
        Estimate traffic density based on the number of vehicles in scene.
        Returns level ('Low', 'Medium', 'High'), badge color, and description.
        """
        if vehicle_count <= self.low_max:
            level = TRAFFIC_DENSITY_LABELS["LOW"]
            desc = "Light traffic flow with high open road capacity"
        elif vehicle_count <= self.medium_max:
            level = TRAFFIC_DENSITY_LABELS["MEDIUM"]
            desc = "Moderate traffic flow, steady vehicle presence"
        else:
            level = TRAFFIC_DENSITY_LABELS["HIGH"]
            desc = "Heavy congestion or high vehicle concentration"

        return {
            "level": level,
            "color": TRAFFIC_DENSITY_COLORS[level],
            "description": desc,
            "vehicle_count": vehicle_count,
        }

    def analyze_frame(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Complete single-frame or image analysis."""
        class_counts = self.count_by_class(detections)
        category_counts = self.get_category_counts(detections)
        density = self.estimate_density(category_counts["vehicles"])

        return {
            "class_counts": class_counts,
            "category_counts": category_counts,
            "density": density,
            "total_detections": len(detections),
        }

    @staticmethod
    def count_unique_tracked(tracked_objects: Set[Tuple[int, str]]) -> Dict[str, Any]:
        """
        Compute unique vehicle and pedestrian counts across video frames.
        tracked_objects contains (track_id, class_name) tuples.
        """
        unique_class_counts: Dict[str, int] = {}
        unique_vehicles = 0
        unique_pedestrians = 0

        for _, cls_name in tracked_objects:
            unique_class_counts[cls_name] = unique_class_counts.get(cls_name, 0) + 1
            if cls_name in VEHICLE_CLASSES:
                unique_vehicles += 1
            elif cls_name in PEDESTRIAN_CLASSES:
                unique_pedestrians += 1

        return {
            "unique_class_counts": unique_class_counts,
            "unique_vehicles": unique_vehicles,
            "unique_pedestrians": unique_pedestrians,
            "unique_total": len(tracked_objects),
        }
