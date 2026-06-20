from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/yolo/data.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    workers=4,
    device="mps"
)