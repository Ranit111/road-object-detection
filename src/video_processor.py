"""Video processing engine with temporal tracking persistence, ROI masking, and high-res inference."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MIN_TRACK_PERSISTENCE,
    VEHICLE_CLASSES,
)
from src.detector import RoadObjectDetector
from src.traffic_analyzer import TrafficAnalyzer
from src.utils import draw_detections

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes video streams or files with YOLO detection, ByteTrack tracking, and temporal filtering."""

    def __init__(
        self,
        detector: RoadObjectDetector,
        traffic_analyzer: Optional[TrafficAnalyzer] = None,
    ) -> None:
        self.detector = detector
        self.analyzer = traffic_analyzer or TrafficAnalyzer()

    def _get_video_writer(
        self,
        output_path: Path,
        fps: float,
        frame_size: Tuple[int, int],
    ) -> cv2.VideoWriter:
        """Initialize VideoWriter with browser-compatible codec fallback."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = frame_size

        codecs = ["mp4v", "avc1", "XVID"]
        writer = None

        for c in codecs:
            fourcc = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if writer.isOpened():
                logger.info("Initialized VideoWriter with codec: %s", c)
                return writer
            writer.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_tracking: bool = True,
        min_track_persistence: int = DEFAULT_MIN_TRACK_PERSISTENCE,
        imgsz: int = DEFAULT_INFERENCE_SIZE,
        roi_top_pct: float = 0.0,
        roi_side_pct: float = 0.0,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[int, int, np.ndarray, Dict[str, Any]], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file frame-by-frame with temporal persistence filtering.

        Args:
            input_path: Path to input video file.
            output_path: Optional path to save processed video.
            conf_threshold: Minimum YOLO confidence threshold.
            iou_threshold: NMS IoU threshold.
            use_tracking: Enable ByteTrack object tracking.
            min_track_persistence: Minimum consecutive frames required to confirm a track (filters flickers).
            imgsz: Inference size (640 or 1280).
            roi_top_pct: Exclude top percentage of frame (e.g. 0.25 for sky/roof cutoff).
            roi_side_pct: Exclude side percentage of frame (vertical lane boundary).
            frame_stride: Process every N-th frame.
            progress_callback: Callback(frame_idx, total_frames, frame_bgr, frame_stats).
            stop_check: Callable returning True if processing should be cancelled early.

        Returns:
            Dictionary with comprehensive video analysis summary.
        """
        in_path = Path(input_path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Video file not found: {input_path}")

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path is not None:
            effective_fps = orig_fps / max(1, frame_stride)
            writer = self._get_video_writer(Path(output_path), effective_fps, (width, height))

        # Temporal filter state
        track_hits: Dict[int, int] = {}
        tracked_unique_objects: Set[Tuple[int, str]] = set()

        frame_history: List[Dict[str, Any]] = []
        cumulative_class_counts: Dict[str, int] = {}
        peak_vehicle_count = 0
        peak_density = "Low"

        processed_frame_count = 0
        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                if stop_check and stop_check():
                    logger.info("Video processing cancelled by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                if frame_stride > 1 and (frame_idx % frame_stride != 0):
                    continue

                frame_t0 = time.time()

                # Inference: tracking or pure detection
                if use_tracking:
                    detections, _ = self.detector.track_frame(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        tracker="bytetrack.yaml",
                        imgsz=imgsz,
                        roi_top_pct=roi_top_pct,
                        roi_side_pct=roi_side_pct,
                    )

                    # Update temporal hit count for each track ID
                    for det in detections:
                        t_id = det.get("track_id")
                        if t_id is not None:
                            track_hits[t_id] = track_hits.get(t_id, 0) + 1

                    # Temporal persistence filter: only keep confirmed detections (>= min_track_persistence)
                    if min_track_persistence > 1:
                        confirmed_detections = [
                            det for det in detections
                            if det.get("track_id") is None or track_hits.get(det["track_id"], 0) >= min_track_persistence
                        ]
                    else:
                        confirmed_detections = detections

                    # Record confirmed unique objects
                    for det in confirmed_detections:
                        t_id = det.get("track_id")
                        if t_id is not None and track_hits.get(t_id, 0) >= min_track_persistence:
                            tracked_unique_objects.add((t_id, det["class_name"]))

                else:
                    detections, _ = self.detector.detect_image(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        imgsz=imgsz,
                        roi_top_pct=roi_top_pct,
                        roi_side_pct=roi_side_pct,
                    )
                    confirmed_detections = detections
                    for cls_name, c in self.analyzer.count_by_class(confirmed_detections).items():
                        cumulative_class_counts[cls_name] = cumulative_class_counts.get(cls_name, 0) + c

                frame_latency = max(time.time() - frame_t0, 1e-5)
                current_fps = round(1.0 / frame_latency, 1)

                # Analyze frame using confirmed detections
                frame_analysis = self.analyzer.analyze_frame(confirmed_detections)
                density_info = frame_analysis["density"]
                vehicles = frame_analysis["category_counts"]["vehicles"]
                pedestrians = frame_analysis["category_counts"]["pedestrians"]

                if vehicles > peak_vehicle_count:
                    peak_vehicle_count = vehicles
                    peak_density = density_info["level"]

                # Annotate frame
                annotated_frame = draw_detections(
                    frame,
                    confirmed_detections,
                    show_labels=True,
                    show_conf=True,
                    show_track_id=use_tracking,
                    density_badge=density_info,
                    roi_top_pct=roi_top_pct,
                    roi_side_pct=roi_side_pct,
                )

                if writer is not None:
                    writer.write(annotated_frame)

                processed_frame_count += 1
                timestamp_sec = round(frame_idx / orig_fps, 2)

                frame_stat = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "fps": current_fps,
                    "vehicles": vehicles,
                    "pedestrians": pedestrians,
                    "density_level": density_info["level"],
                    "detections_count": len(confirmed_detections),
                }
                frame_history.append(frame_stat)

                if progress_callback:
                    progress_callback(frame_idx, total_frames, annotated_frame, frame_stat)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        total_elapsed = max(time.time() - start_time, 1e-5)
        avg_fps = round(processed_frame_count / total_elapsed, 1) if processed_frame_count > 0 else 0.0

        if use_tracking:
            tracked_summary = self.analyzer.count_unique_tracked(tracked_unique_objects)
            final_class_counts = tracked_summary["unique_class_counts"]
        else:
            final_class_counts = cumulative_class_counts

        return {
            "total_frames_analyzed": processed_frame_count,
            "total_frames_in_video": total_frames,
            "elapsed_seconds": round(total_elapsed, 2),
            "avg_fps": avg_fps,
            "use_tracking": use_tracking,
            "min_track_persistence": min_track_persistence,
            "peak_vehicle_count": peak_vehicle_count,
            "peak_density": peak_density,
            "final_class_counts": final_class_counts,
            "unique_tracked_total": len(tracked_unique_objects) if use_tracking else None,
            "frame_history": frame_history,
            "output_video_path": str(output_path) if output_path else None,
        }
