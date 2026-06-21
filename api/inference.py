from ultralytics import YOLO
from collections import Counter
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from api.metrics import (
    REQUEST_COUNT,
    INFERENCE_LATENCY,
    DETECTIONS_TOTAL,
    AVG_CONFIDENCE
)
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer
import av

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/best.pt"
MODEL_VERSION = "yolov8n_baseline_v1"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "car",
    "person",
    "traffic light",
    "traffic sign",
    "bike",
    "bus",
    "truck",
    "rider",
]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_DIR/"inference.log",
    level=logging.INFO,
    format="%(message)s"
)

# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")

# ============================================================
# INFERENCE
# ============================================================

def predict(image_path):

    start_time = time.perf_counter()

    results = model.predict(
        source=image_path,
        conf=0.40,
        verbose=False
    )

    detections = []
    class_counter = Counter()

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = CLASS_NAMES[cls_id]

            confidence = round(
                float(box.conf[0]),
                4
            )

            avg_conf = 0

            if detections:
                avg_conf =sum(
                    d["confidence"]                    
                    for d in detections
                ) / len(detections)
            AVG_CONFIDENCE.set(avg_conf)

            bbox = [
                round(x, 2)
                for x in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                }
            )

            class_counter[class_name] += 1

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    REQUEST_COUNT.inc()
    INFERENCE_LATENCY.observe(latency_ms)
    DETECTIONS_TOTAL.inc(len(detections))

    log_record ={
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "num_detections": len(detections),
        "avg_confidence": round(avg_conf, 4),
        "class_distribution": dict(class_counter)
    }

    logging.info(
        json.dumps(log_record)
    )

    return {
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "num_detections": len(detections),
        "class_distribution": dict(class_counter),
        "detections": detections
    }

def predict_with_image(image_path):

    results = model.predict(
        source=image_path,
        conf=0.40,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    Image.fromarray(annotated).save(
        temp_file.name
    )

    detections = []

    for box in result.boxes:

        cls_id = int(box.cls[0])

        detections.append(
            {
                "class": CLASS_NAMES[cls_id],
                "confidence": round(
                    float(box.conf[0]),
                    4
                ),
                "bbox": [
                    round(x, 2)
                    for x in box.xyxy[0].tolist()
                ]
            }
        )

    return detections, temp_file.name


def predict_video(video_path):

    results = model.predict(
        source=video_path,
        conf=0.4,
        save=True,
        project="runs/detect",
        name="video_predict",
        exist_ok=True,
        verbose=False,
        stream=True
    )

    for _ in results:
        pass

    video_files = [
        p for p in Path(".").rglob("*.mp4")
        if "video_predict" in str(p)
    ]

    if not video_files:
        raise FileNotFoundError(
            "No output video found."
        )

    latest_video = max(
        video_files,
        key=lambda x: x.stat().st_mtime
    )

    return str(latest_video)


class VideoProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        results = model(
            img,
            conf=0.4
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )