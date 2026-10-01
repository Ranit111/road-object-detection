"""YOLO Road Scene Object Detector module.

Supports scalable models (yolov8n, yolov8s, yolov8m, custom fine-tuned weights),
high-resolution inference (imgsz=640/1280), Road ROI masking, class-specific confidence,
and post-NMS deduplication.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

# Optimize CPU threads for fast inference on multi-core systems
try:
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, os.cpu_count() or 4))
except Exception:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.config import (
    CLASS_CONFIDENCE_THRESHOLDS,
    CLASS_GEOMETRY_CONSTRAINTS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    MIN_BBOX_AREA,
    MIN_BBOX_HEIGHT,
    MIN_BBOX_WIDTH,
    ROAD_CLASSES,
    VEHICLE_CLASSES,
)

logger = logging.getLogger(__name__)


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_box_containment(box1: List[float], box2: List[float]) -> float:
    """Compute intersection over the smaller box's area to catch nested phantom boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    smaller_area = min(area1, area2)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def apply_post_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Remove duplicate or heavily overlapping detections.
    Retains highest confidence detections and suppresses redundant nested boxes.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in sorted_dets:
        box = det["bbox_xyxy"]
        cls_name = det["class_name"]

        is_duplicate = False
        for k in kept:
            k_box = k["bbox_xyxy"]
            k_cls = k["class_name"]

            iou = compute_box_iou(box, k_box)
            containment = compute_box_containment(box, k_box)

            if iou > iou_threshold:
                is_duplicate = True
                break

            if containment > containment_threshold:
                if cls_name == k_cls or (cls_name in VEHICLE_CLASSES and k_cls in VEHICLE_CLASSES):
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept.append(det)

    return kept


def is_valid_detection_geometry(
    class_name: str,
    bbox: List[float],
    min_width: int = MIN_BBOX_WIDTH,
    min_height: int = MIN_BBOX_HEIGHT,
    min_area: int = MIN_BBOX_AREA,
) -> bool:
    """
    Validate bounding box geometry against realistic road object proportions.
    Filters out background noise such as vertical street poles, road lane lines, and manholes.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    area = w * h

    if w < min_width or h < min_height or area < min_area:
        return False

    constraints = CLASS_GEOMETRY_CONSTRAINTS.get(class_name)
    if constraints:
        c_min_w = constraints.get("min_width", min_width)
        c_min_h = constraints.get("min_height", min_height)
        if w < c_min_w or h < c_min_h:
            return False

        aspect = w / max(1e-4, h)
        min_aspect = constraints.get("min_aspect_ratio", 0.15)
        max_aspect = constraints.get("max_aspect_ratio", 5.5)
        if aspect < min_aspect or aspect > max_aspect:
            return False

    return True


class RoadObjectDetector:
    """Wrapper around Ultralytics YOLO models with dynamic scaling and road optimizations."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        target_classes: Optional[Dict[int, str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.target_classes = target_classes or ROAD_CLASSES
        self._target_class_ids = list(self.target_classes.keys())
        self.device = 0 if (torch and torch.cuda.is_available()) else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model weights, downloading if necessary."""
        global YOLO
        if YOLO is None:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO

        candidate_paths = [
            self.model_path,
            self.model_path.resolve(),
            Path("models") / self.model_path.name,
            Path(self.model_path.name),
        ]
        chosen_path = None
        for p in candidate_paths:
            if p.is_file():
                chosen_path = str(p)
                break

        if chosen_path:
            self.model = YOLO(chosen_path)
            logger.info("YOLO loaded from: %s", chosen_path)
        else:
            model_target = self.model_path.name if self.model_path.name.endswith(".pt") else "yolov8n.pt"
            logger.info("Local model not found. Ultralytics loading/downloading: %s...", model_target)
            self.model = YOLO(model_target)
            # Cache to models dir if possible
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        logger.info("YOLO model initialized successfully.")

    def set_model(self, model_name_or_path: Union[str, Path]) -> None:
        """Dynamically switch model (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt, or custom weights)."""
        new_path = Path(model_name_or_path)
        if self.model_path.name != new_path.name or self.model is None:
            self.model_path = new_path
            self._load_model()

    def _parse_results(
        self,
        result: Any,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_conf_thresholds: Optional[Dict[str, float]] = None,
        min_width: int = MIN_BBOX_WIDTH,
        min_height: int = MIN_BBOX_HEIGHT,
        min_area: int = MIN_BBOX_AREA,
        filter_geometry: bool = True,
        apply_nms: bool = True,
        nms_iou: float = DEFAULT_IOU_THRESHOLD,
        roi_top_pct: float = 0.0,
        roi_side_pct: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Parse, filter, and deduplicate Ultralytics detection outputs."""
        detections: List[Dict[str, Any]] = []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        track_ids = None
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        # Image dimensions for ROI filtering
        img_h = result.orig_shape[0] if hasattr(result, "orig_shape") else 0
        img_w = result.orig_shape[1] if hasattr(result, "orig_shape") else 0
        roi_cutoff_y = img_h * roi_top_pct if (roi_top_pct > 0.0 and img_h > 0) else 0.0
        roi_cutoff_x = img_w * roi_side_pct if (roi_side_pct > 0.0 and img_w > 0) else 0.0

        thresh_map = class_conf_thresholds if class_conf_thresholds is not None else CLASS_CONFIDENCE_THRESHOLDS

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            if self._target_class_ids and cls_id not in self._target_class_ids:
                continue

            cls_name = self.target_classes.get(cls_id, result.names.get(cls_id, f"class_{cls_id}"))
            conf = float(confs[i])
            bbox = [float(coord) for coord in xyxy[i]]
            track_id = int(track_ids[i]) if track_ids is not None else None

            # 1. Road ROI filter: horizontal cutoff (sky/roof) and vertical boundary (sides)
            if roi_cutoff_y > 0:
                y_center = (bbox[1] + bbox[3]) / 2.0
                if y_center < roi_cutoff_y or bbox[3] < roi_cutoff_y:
                    continue

            if roi_cutoff_x > 0:
                x_center = (bbox[0] + bbox[2]) / 2.0
                if x_center < roi_cutoff_x or x_center > (img_w - roi_cutoff_x):
                    continue

            # 2. Class-specific confidence threshold check
            class_req_conf = thresh_map.get(cls_name, conf_threshold)
            min_conf = max(conf_threshold, class_req_conf) if conf_threshold > 0.30 else class_req_conf
            if conf < min_conf:
                continue

            # 3. Minimum bounding box size & geometry checks (manholes, road markings, poles)
            if filter_geometry and not is_valid_detection_geometry(
                cls_name, bbox, min_width=min_width, min_height=min_height, min_area=min_area
            ):
                continue

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": bbox,
                "track_id": track_id,
            })

        # 4. Post-NMS deduplication to remove duplicate/nested boxes
        if apply_nms:
            detections = apply_post_nms(detections, iou_threshold=nms_iou)

        return detections

    def detect_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        class_conf_thresholds: Optional[Dict[str, float]] = None,
        filter_geometry: bool = True,
        apply_nms: bool = True,
        imgsz: int = DEFAULT_INFERENCE_SIZE,
        roi_top_pct: float = 0.0,
        roi_side_pct: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Run YOLO object detection on a single image.

        Args:
            image_input: File path, numpy array (BGR/RGB), or PIL Image.
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            class_conf_thresholds: Optional custom per-class confidence thresholds.
            filter_geometry: Enable size and aspect ratio filtering.
            apply_nms: Enable post-process NMS deduplication.
            imgsz: Inference image size (640 or 1280).
            roi_top_pct: Road ROI horizontal cutoff ratio (sky/horizon).
            roi_side_pct: Road ROI vertical boundary ratio (sides/sidewalks).
        """
        if self.model is None:
            self._load_model()

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty.")
        elif isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {image_input}")

        yolo_conf = min(conf_threshold, 0.40)

        results = self.model.predict(
            source=image_input,
            conf=yolo_conf,
            iou=iou_threshold,
            classes=self._target_class_ids,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        first_result = results[0] if results else None
        detections = self._parse_results(
            first_result,
            conf_threshold=conf_threshold,
            class_conf_thresholds=class_conf_thresholds,
            filter_geometry=filter_geometry,
            apply_nms=apply_nms,
            nms_iou=iou_threshold,
            roi_top_pct=roi_top_pct,
            roi_side_pct=roi_side_pct,
        )
        return detections, first_result

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
        class_conf_thresholds: Optional[Dict[str, float]] = None,
        filter_geometry: bool = True,
        apply_nms: bool = True,
        imgsz: int = DEFAULT_INFERENCE_SIZE,
        roi_top_pct: float = 0.0,
        roi_side_pct: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Run YOLO object tracking on a video frame using ByteTrack.

        Args:
            frame: Video frame as numpy array (BGR).
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            tracker: Tracker config file (default: 'bytetrack.yaml').
            persist: Keep track history between consecutive frames.
            imgsz: Inference size (640 or 1280).
            roi_top_pct: Road ROI top cutoff ratio.
            roi_side_pct: Road ROI vertical boundary ratio.
        """
        if self.model is None:
            self._load_model()

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is None or empty.")

        yolo_conf = min(conf_threshold, 0.40)

        results = self.model.track(
            source=frame,
            persist=persist,
            conf=yolo_conf,
            iou=iou_threshold,
            classes=self._target_class_ids,
            tracker=tracker,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        first_result = results[0] if results else None
        detections = self._parse_results(
            first_result,
            conf_threshold=conf_threshold,
            class_conf_thresholds=class_conf_thresholds,
            filter_geometry=filter_geometry,
            apply_nms=apply_nms,
            nms_iou=iou_threshold,
            roi_top_pct=roi_top_pct,
            roi_side_pct=roi_side_pct,
        )
        return detections, first_result
