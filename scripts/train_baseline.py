from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/yolo/data.yaml",
    epochs=100,
    patience=15,
    imgsz=640,
    batch=16,
    workers=4,
    device="mps",
    project="runs/perception_ops",
    name="yolov8n_baseline"
)