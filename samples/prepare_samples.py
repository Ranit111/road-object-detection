"""Script to prepare sample road image and test video for demonstration and testing."""

from pathlib import Path
import urllib.request
import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE_PATH = SAMPLES_DIR / "sample_road.jpg"
SAMPLE_VIDEO_PATH = SAMPLES_DIR / "sample_traffic.mp4"


def download_or_create_sample_image() -> Path:
    """Download standard road/street scene or generate synthetic road scene."""
    if SAMPLE_IMAGE_PATH.exists():
        return SAMPLE_IMAGE_PATH

    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    try:
        print(f"Downloading sample road scene from {url}...")
        urllib.request.urlretrieve(url, str(SAMPLE_IMAGE_PATH))
        print(f"Sample image saved to {SAMPLE_IMAGE_PATH}")
        return SAMPLE_IMAGE_PATH
    except Exception as e:
        print(f"Network download failed: {e}. Generating synthetic road scene...")
        # Create a synthetic road scene
        h, w = 480, 640
        img = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)  # gray background
        # Road asphalt
        cv2.rectangle(img, (0, 200), (w, h), (50, 50, 50), -1)
        # Lane markings
        for x in range(0, w, 80):
            cv2.rectangle(img, (x, 340), (x + 40, 345), (255, 255, 255), -1)
        # Sky
        cv2.rectangle(img, (0, 0), (w, 200), (220, 190, 140), -1)
        cv2.imwrite(str(SAMPLE_IMAGE_PATH), img)
        return SAMPLE_IMAGE_PATH


def create_sample_video_from_image(image_path: Path, output_video_path: Path, num_frames: int = 45) -> Path:
    """Create a smooth panning video from sample road image for testing video pipeline."""
    if output_video_path.exists():
        return output_video_path

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    h, w = img.shape[:2]
    # Crop window that pans slightly horizontally
    crop_w = int(w * 0.85)
    crop_h = int(h * 0.85)
    max_dx = w - crop_w
    max_dy = h - crop_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15.0
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (crop_w, crop_h))

    for i in range(num_frames):
        alpha = i / max(1, num_frames - 1)
        x_start = int(alpha * max_dx)
        y_start = int((1.0 - alpha) * max_dy)
        frame = img[y_start : y_start + crop_h, x_start : x_start + crop_w]
        writer.write(frame)

    writer.release()
    print(f"Sample video created at {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    img_path = download_or_create_sample_image()
    create_sample_video_from_image(img_path, SAMPLE_VIDEO_PATH)
