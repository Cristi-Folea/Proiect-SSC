import torch 
import onnx 
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

device = 0 if torch.cuda.is_available() else "cpu"
print(device)

results = model.predict(
    source="../../data/bus.jpg",
    device=device,
    imgsz=640,
    conf=0.25,
    save=True
)
