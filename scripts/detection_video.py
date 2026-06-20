from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/perception_ops/yolov8n_baseline/weights/best.pt"
)

model.track(
    source="data/test_video/",
    tracker="bytetrack.yaml",
    conf=0.25,
    save=True
)