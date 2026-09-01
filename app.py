"""Streamlit application for real-time object detection and face verification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from face_utils import is_reference_person


MODEL_DIR = Path(__file__).parent / "models"
SSD_PROTOTXT = MODEL_DIR / "deploy.prototxt"
SSD_WEIGHTS = MODEL_DIR / "mobilenet_iter_73000.caffemodel"

SSD_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@st.cache_resource
def load_yolo_model() -> YOLO:
    """Load YOLO once per Streamlit session."""
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_ssd_model() -> cv2.dnn_Net:
    """Load the MobileNet SSD network from the local models directory."""
    return cv2.dnn.readNetFromCaffe(str(SSD_PROTOTXT), str(SSD_WEIGHTS))


def decode_uploaded_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """Decode an uploaded image without writing it to disk."""
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a supported image.")
    return image


def draw_yolo_detections(
    frame: np.ndarray,
    result,
    reference_image: np.ndarray,
    frame_number: int,
    verification_cache: dict[int, bool],
    verification_interval: int,
) -> tuple[np.ndarray, set[int]]:
    """Draw YOLO detections and verify tracked people independently."""
    annotated = frame.copy()
    visible_tracks: set[int] = set()

    for fallback_id, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        label = result.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        display = f"YOLO: {label}"
        color = (0, 255, 255)

        if label == "person":
            track_id = int(box.id.item()) if box.id is not None else fallback_id
            visible_tracks.add(track_id)
            person_crop = frame[y1:y2, x1:x2]

            should_verify = (
                person_crop.size > 0
                and (track_id not in verification_cache or frame_number % verification_interval == 0)
            )
            if should_verify:
                verification_cache[track_id] = is_reference_person(person_crop, reference_image)

            if verification_cache.get(track_id, False):
                display = "Reference Person"
                color = (0, 200, 0)
            else:
                display = "Unknown Person"
                color = (0, 0, 255)
        elif label == "cat":
            color = (255, 0, 0)
        elif label == "dog":
            color = (255, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            display,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    return annotated, visible_tracks


def draw_ssd_detections(frame: np.ndarray, ssd_net: cv2.dnn_Net, threshold: float) -> np.ndarray:
    """Run MobileNet SSD and draw detections on a copy of the frame."""
    annotated = frame.copy()
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    ssd_net.setInput(blob)
    detections = ssd_net.forward()
    height, width = frame.shape[:2]

    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2])
        if confidence < threshold:
            continue

        class_id = int(detections[0, 0, index, 1])
        if not 0 <= class_id < len(SSD_CLASSES):
            continue

        x1, y1, x2, y2 = (
            detections[0, 0, index, 3:7] * np.array([width, height, width, height])
        ).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        label = f"SSD: {SSD_CLASSES[class_id]} {confidence:.0%}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 0),
            2,
        )

    return annotated


def run_camera(reference_image: np.ndarray, confidence_threshold: float, interval: int) -> None:
    """Open the local webcam and stream annotated frames to the app."""
    if not SSD_PROTOTXT.exists() or not SSD_WEIGHTS.exists():
        st.error("MobileNet SSD files are missing. Run `python download_models.py` first.")
        return

    yolo_model = load_yolo_model()
    ssd_net = load_ssd_model()
    left_column, right_column = st.columns(2)
    yolo_window = left_column.empty()
    ssd_window = right_column.empty()
    left_column.caption("YOLOv8 + face verification")
    right_column.caption("MobileNet SSD")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("The webcam could not be opened. Check camera permissions and try again.")
        return

    frame_number = 0
    verification_cache: dict[int, bool] = {}

    try:
        while True:
            success, frame = camera.read()
            if not success:
                st.error("A frame could not be read from the webcam.")
                break

            frame_number += 1
            yolo_result = yolo_model.track(frame, persist=True, verbose=False)[0]
            yolo_frame, visible_tracks = draw_yolo_detections(
                frame,
                yolo_result,
                reference_image,
                frame_number,
                verification_cache,
                interval,
            )
            verification_cache = {
                track_id: verified
                for track_id, verified in verification_cache.items()
                if track_id in visible_tracks
            }
            ssd_frame = draw_ssd_detections(frame, ssd_net, confidence_threshold)

            yolo_window.image(cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            ssd_window.image(cv2.cvtColor(ssd_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    finally:
        camera.release()


def main() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(
        page_title="Object Detection & Face Verification",
        page_icon="🎥",
        layout="wide",
    )
    st.title("Real-Time Object Detection & Face Verification")
    st.write(
        "Compare YOLOv8 and MobileNet SSD detections while verifying each tracked "
        "person against a reference image."
    )

    with st.sidebar:
        st.header("Settings")
        uploaded_reference = st.file_uploader(
            "Reference face image",
            type=("jpg", "jpeg", "png"),
            help="The image is decoded in memory and is never written to the project directory.",
        )
        confidence_threshold = st.slider(
            "MobileNet confidence threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )
        verification_interval = st.slider(
            "Face verification interval (frames)",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
        )
        start_camera = st.toggle("Start camera", value=False)

    if uploaded_reference is None:
        st.info("Upload a reference face image to enable the camera.")
        return

    try:
        reference_image = decode_uploaded_image(uploaded_reference)
    except ValueError as error:
        st.error(str(error))
        return

    st.sidebar.image(
        cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB),
        caption="Reference image (memory only)",
    )

    if start_camera:
        run_camera(reference_image, confidence_threshold, verification_interval)


if __name__ == "__main__":
    main()
