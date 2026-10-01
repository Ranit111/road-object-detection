"""Smart Road Scene Analyzer - Streamlit Dashboard.

Clean, professional computer-vision dashboard for road scene object detection,
vehicle/pedestrian counting, and traffic density estimation.
Automatic high-accuracy model & resolution for images, streamlined video models,
and dual horizontal/vertical Road ROI controls.
"""

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.config import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_MODEL_PATH,
    IMAGE_DEFAULT_MODEL,
    PEDESTRIAN_CLASSES,
    ROAD_CLASSES,
    TRAFFIC_DENSITY_COLORS,
    TRAFFIC_DENSITY_THRESHOLDS,
    VEHICLE_CLASSES,
    VIDEO_MODELS,
)
from src.detector import RoadObjectDetector
from src.traffic_analyzer import TrafficAnalyzer
from src.utils import (
    draw_detections,
    export_detections_to_dataframe,
    export_summary_to_csv,
    save_annotated_image,
)
from src.video_processor import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Smart Road Scene Analyzer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS matching DESIGN.md: clean, light neutral, dark text, subtle accent
st.markdown(
    """
    <style>
    .main {
        background-color: #FAFAFA;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
    }
    .density-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        color: #FFFFFF;
    }
    .status-note {
        font-size: 12px;
        color: #6B7280;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO model weights...")
def get_detector(model_target: str) -> RoadObjectDetector:
    """Load and cache the YOLO detector instance for the selected model variant."""
    return RoadObjectDetector(model_path=model_target, target_classes=ROAD_CLASSES)


def main():
    # Header
    st.title("Smart Road Scene Analyzer")
    st.caption("YOLO road scene object detection, counting, and traffic density analysis.")

    # Sidebar: configuration & controls
    with st.sidebar:
        st.subheader("Controls & Settings")

        app_mode = st.radio(
            "Detection Mode",
            ["Image Detection", "Video Detection"],
            index=0,
        )

        # High resolution standard (1280px) preset in backend for all modes
        imgsz = DEFAULT_INFERENCE_SIZE

        # Model configuration per mode
        if app_mode == "Image Detection":
            # Automatic High-Precision Model (yolov8m.pt) without cluttering UI
            target_model_file = IMAGE_DEFAULT_MODEL
        else:
            # Video Detection: 2 streamlined options (YOLOv8n vs YOLOv8m)
            st.markdown("---")
            st.subheader("Model Selection")
            video_model_choice = st.selectbox(
                "YOLO Model Variant",
                list(VIDEO_MODELS.keys()),
                index=0,
                help="Choose YOLOv8n for fast processing or YOLOv8m for high accuracy.",
            )
            target_model_file = VIDEO_MODELS[video_model_choice]

        # Road Region of Interest (ROI) with Vertical and Horizon Controls
        st.markdown("---")
        st.subheader("Road Region of Interest (ROI)")
        enable_roi = st.checkbox(
            "Enable Road ROI Masking",
            value=False,
            help="Restricts detection to the road surface, ignoring the sky, tree line, or side pavements.",
        )
        roi_top_pct = 0.0
        roi_side_pct = 0.0
        if enable_roi:
            roi_top_pct = st.slider(
                "Horizon Cutoff % (Top)",
                min_value=0.05,
                max_value=0.50,
                value=0.25,
                step=0.05,
                help="Exclude sky, clouds, and tall roofs above the road horizon.",
            )
            roi_side_pct = st.slider(
                "Vertical Lane Boundary % (Sides)",
                min_value=0.0,
                max_value=0.35,
                value=0.0,
                step=0.05,
                help="Exclude side walkways/pavements outside the vertical road corridor.",
            )

        st.markdown("---")
        st.subheader("Inference Parameters")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.20,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Minimum confidence score for detected road objects (tuned to 0.50 to suppress false positives).",
        )

        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.20,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Non-Maximum Suppression threshold to eliminate duplicate boxes.",
        )

        # Class filtering
        all_class_names = list(ROAD_CLASSES.values())
        selected_classes = st.multiselect(
            "Filter Object Classes",
            options=all_class_names,
            default=["car", "person"],
            help="Select which road object categories to detect and count.",
        )
        st.caption("🔍 False-positive filters: Car ≥ 0.50 | Person ≥ 0.45 | Min-size & NMS active")

        use_tracking = False
        min_track_persistence = 1
        # Fast processing default (frame_stride=2) in backend without UI slider
        frame_stride = DEFAULT_FRAME_STRIDE

        if app_mode == "Video Detection":
            st.markdown("---")
            st.subheader("Video Tracking & Temporal Filtering")
            use_tracking = st.checkbox(
                "Enable ByteTrack Tracking",
                value=True,
                help="Tracks individual objects across frames to compute unique vehicle/pedestrian counts.",
            )
            if use_tracking:
                min_track_persistence = st.slider(
                    "Temporal Persistence Filter (Frames)",
                    min_value=1,
                    max_value=8,
                    value=3,
                    help="Object must persist across at least N frames to be confirmed. Eliminates flickering shadows, manholes, and glare.",
                )

        st.markdown("---")
        st.caption(
            f"Traffic Density Rules: Low (<= {TRAFFIC_DENSITY_THRESHOLDS['low_max']}), "
            f"Medium (4-{TRAFFIC_DENSITY_THRESHOLDS['medium_max']}), "
            f"High (>={TRAFFIC_DENSITY_THRESHOLDS['medium_max'] + 1} vehicles)"
        )

    # Initialize model
    try:
        detector = get_detector(target_model_file)
        analyzer = TrafficAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize YOLO detector: {e}")
        st.stop()

    # Map selected classes to class IDs
    name_to_id = {v: k for k, v in ROAD_CLASSES.items()}
    active_class_ids = [name_to_id[name] for name in selected_classes if name in name_to_id]
    detector.target_classes = {k: ROAD_CLASSES[k] for k in active_class_ids}
    detector._target_class_ids = active_class_ids

    # Route based on selected mode
    if app_mode == "Image Detection":
        render_image_mode(detector, analyzer, conf_threshold, iou_threshold, imgsz, roi_top_pct, roi_side_pct)
    else:
        render_video_mode(
            detector,
            analyzer,
            conf_threshold,
            iou_threshold,
            use_tracking,
            min_track_persistence,
            imgsz,
            roi_top_pct,
            roi_side_pct,
            frame_stride,
        )


def render_image_mode(
    detector: RoadObjectDetector,
    analyzer: TrafficAnalyzer,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    roi_top_pct: float,
    roi_side_pct: float,
):
    """Render Image Detection UI, preview, metrics, and downloads."""
    st.subheader("Image Analysis")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a road scene image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.checkbox("Use Demo Sample", value=False, help="Load the bundled road scene sample")

    sample_path = Path("samples/sample_road.jpg")
    image_bgr = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif use_sample and sample_path.exists():
        image_bgr = cv2.imread(str(sample_path))
    else:
        st.info("Please upload a road image or check 'Use Demo Sample' to begin detection.")
        return

    if image_bgr is None:
        st.error("Invalid or corrupted image file. Please upload a valid image.")
        return

    # Run detection
    t_start = time.time()
    try:
        detections, _ = detector.detect_image(
            image_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            roi_top_pct=roi_top_pct,
            roi_side_pct=roi_side_pct,
        )
    except Exception as e:
        st.error(f"Error during image detection: {e}")
        return
    inference_time_ms = round((time.time() - t_start) * 1000, 1)

    # Traffic and counting analysis
    analysis = analyzer.analyze_frame(detections)
    cat_counts = analysis["category_counts"]
    density = analysis["density"]
    class_counts = analysis["class_counts"]

    # Annotate image
    annotated_bgr = draw_detections(
        image_bgr,
        detections,
        show_labels=True,
        show_conf=True,
        show_track_id=False,
        density_badge=density,
        roi_top_pct=roi_top_pct,
        roi_side_pct=roi_side_pct,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display Metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Vehicles</div>
                <div class="metric-value">{cat_counts['vehicles']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Pedestrians</div>
                <div class="metric-value">{cat_counts['pedestrians']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Objects</div>
                <div class="metric-value">{cat_counts['total']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        level_color = density["color"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Traffic Density</div>
                <div>
                    <span class="density-badge" style="background-color: {level_color};">
                        {density['level']}
                    </span>
                </div>
                <div class="status-note">{density['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(f"Inference Latency: {inference_time_ms} ms | Detections: {len(detections)} | Auto Model: {detector.model_path.name} | Auto Res: {imgsz}px")

    # Image Preview (Side by side)
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original Image**")
        st.image(original_rgb, use_container_width=True)
    with img_col2:
        st.markdown("**Processed Detection**")
        st.image(annotated_rgb, use_container_width=True)

    # Tabular Breakdown & Export
    st.markdown("### Detection Details")
    tab1, tab2 = st.tabs(["Class Breakdown", "Object Bounding Boxes"])

    with tab1:
        if class_counts:
            df_counts = pd.DataFrame(
                list(class_counts.items()),
                columns=["Object Class", "Count"],
            ).sort_values(by="Count", ascending=False)
            st.dataframe(df_counts, use_container_width=True, hide_index=True)
        else:
            st.write("No road objects detected with current thresholds.")

    with tab2:
        if detections:
            df_det = export_detections_to_dataframe(detections)
            st.dataframe(df_det, use_container_width=True, hide_index=True)
        else:
            st.write("No bounding boxes available.")

    # Downloads section
    st.markdown("### Download Results")
    down_col1, down_col2 = st.columns(2)

    with down_col1:
        success, enc_img = cv2.imencode(".jpg", annotated_bgr)
        if success:
            st.download_button(
                label="📥 Download Annotated Image (JPG)",
                data=enc_img.tobytes(),
                file_name="road_scene_detected.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

    with down_col2:
        summary_csv = io.StringIO()
        df_summary = pd.DataFrame([
            {"Metric": "Vehicles Count", "Value": cat_counts["vehicles"]},
            {"Metric": "Pedestrians Count", "Value": cat_counts["pedestrians"]},
            {"Metric": "Total Objects", "Value": cat_counts["total"]},
            {"Metric": "Traffic Density", "Value": density["level"]},
            {"Metric": "Inference Latency (ms)", "Value": inference_time_ms},
            {"Metric": "Model Variant", "Value": detector.model_path.name},
            {"Metric": "Inference Resolution", "Value": f"{imgsz}px"},
            *[{"Metric": f"Count ({cls})", "Value": cnt} for cls, cnt in class_counts.items()],
        ])
        df_summary.to_csv(summary_csv, index=False)
        st.download_button(
            label="📥 Download Detection Summary (CSV)",
            data=summary_csv.getvalue(),
            file_name="road_scene_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_video_mode(
    detector: RoadObjectDetector,
    analyzer: TrafficAnalyzer,
    conf_threshold: float,
    iou_threshold: float,
    use_tracking: bool,
    min_track_persistence: int,
    imgsz: int,
    roi_top_pct: float,
    roi_side_pct: float,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
):
    """Render Video Detection UI, real-time progress, live FPS, and video export."""
    st.subheader("Video Analysis")

    col_vid_up, col_vid_demo = st.columns([3, 1])
    with col_vid_up:
        uploaded_video = st.file_uploader(
            "Upload a road scene video (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
        )
    with col_vid_demo:
        st.write("")
        st.write("")
        use_sample_video = st.checkbox("Use Demo Video", value=False, help="Load the bundled road traffic video sample")

    sample_vid_path = Path("samples/sample_traffic.mp4")
    temp_in_path = None
    is_temp_file = False

    if uploaded_video is not None:
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_in.write(uploaded_video.read())
        temp_in_path = temp_in.name
        temp_in.close()
        is_temp_file = True
    elif use_sample_video and sample_vid_path.exists():
        temp_in_path = str(sample_vid_path)
        is_temp_file = False
    else:
        st.info("Please upload a road video or check 'Use Demo Video' to begin video detection.")
        return

    temp_out_path = str(Path(temp_in_path).parent / f"analyzed_{Path(temp_in_path).stem}.mp4")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        start_processing = st.button("▶ Start Video Analysis", use_container_width=True, type="primary")

    if not start_processing:
        st.caption("Click 'Start Video Analysis' to begin fast frame-by-frame processing.")
        return

    # Layout for real-time progress & video preview
    progress_bar = st.progress(0, text="Initializing video processor...")
    live_status_col1, live_status_col2, live_status_col3, live_status_col4 = st.columns(4)
    fps_placeholder = live_status_col1.empty()
    vehicles_placeholder = live_status_col2.empty()
    peds_placeholder = live_status_col3.empty()
    density_placeholder = live_status_col4.empty()

    frame_preview_placeholder = st.empty()

    video_proc = VideoProcessor(detector=detector, traffic_analyzer=analyzer)

    def update_live_progress(frame_idx: int, total_frames: int, frame_bgr: np.ndarray, stat: Dict):
        pct = min(1.0, frame_idx / max(1, total_frames))
        progress_bar.progress(pct, text=f"Processing frame {frame_idx} of {total_frames}...")

        fps_placeholder.metric("Current FPS", f"{stat['fps']:.1f}")
        vehicles_placeholder.metric("Vehicles (Frame)", stat["vehicles"])
        peds_placeholder.metric("Pedestrians (Frame)", stat["pedestrians"])
        density_placeholder.metric("Density (Frame)", stat["density_level"])

        if frame_idx % 3 == 0 or frame_idx == total_frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_preview_placeholder.image(frame_rgb, caption=f"Frame {frame_idx} | {stat['fps']} FPS", use_container_width=True)

    with st.spinner("Processing video..."):
        try:
            summary = video_proc.process_video(
                input_path=temp_in_path,
                output_path=temp_out_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_tracking=use_tracking,
                min_track_persistence=min_track_persistence,
                imgsz=imgsz,
                roi_top_pct=roi_top_pct,
                roi_side_pct=roi_side_pct,
                frame_stride=frame_stride,
                progress_callback=update_live_progress,
            )
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            return
        finally:
            if is_temp_file and temp_in_path and os.path.exists(temp_in_path):
                try:
                    os.remove(temp_in_path)
                except Exception:
                    pass

    progress_bar.progress(1.0, text="Video processing complete!")
    st.success("Video analysis completed successfully.")

    # Final summary metrics
    st.markdown("### Video Summary Statistics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Frames Analyzed</div>
                <div class="metric-value">{summary['total_frames_analyzed']}</div>
                <div class="status-note">{summary['elapsed_seconds']}s elapsed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average FPS</div>
                <div class="metric-value">{summary['avg_fps']}</div>
                <div class="status-note">Video pipeline speed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        tracked_info = summary.get("unique_tracked_total")
        val = tracked_info if tracked_info is not None else summary["peak_vehicle_count"]
        title = "Unique Confirmed Objects" if use_tracking else "Peak Vehicles in Scene"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        p_density = summary["peak_density"]
        p_color = TRAFFIC_DENSITY_COLORS.get(p_density, "#16A34A")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Peak Traffic Density</div>
                <div>
                    <span class="density-badge" style="background-color: {p_color};">
                        {p_density}
                    </span>
                </div>
                <div class="status-note">Highest congestion observed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Class breakdown
    st.markdown("### Object Counts")
    if summary["final_class_counts"]:
        col_name = "Count (" + ("Unique Confirmed" if use_tracking else "Cumulative") + ")"
        df_classes = pd.DataFrame(
            list(summary["final_class_counts"].items()),
            columns=["Object Class", col_name],
        )
        df_classes = df_classes.sort_values(by=col_name, ascending=False)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("### Download Processed Video & Summary")
    down1, down2 = st.columns(2)

    with down1:
        if os.path.exists(temp_out_path):
            with open(temp_out_path, "rb") as vf:
                video_bytes = vf.read()
            st.download_button(
                label="📥 Download Annotated Video (MP4)",
                data=video_bytes,
                file_name="road_scene_analyzed.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    with down2:
        if summary.get("frame_history"):
            df_history = pd.DataFrame(summary["frame_history"])
            csv_data = df_history.to_csv(index=False)
            st.download_button(
                label="📥 Download Frame History Log (CSV)",
                data=csv_data,
                file_name="traffic_density_frame_log.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
