from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/perception_ops/yolov8n_baseline/weights/best.pt"
)

results = model.predict(
    source="data/test_video/",
    conf=0.25,
    save=True
)