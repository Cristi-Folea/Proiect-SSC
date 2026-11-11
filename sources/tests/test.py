import torch 
import onnx 
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(
    format = "onnx",
    imgsz = 640,
    batch = 1,
    dynamic = False,
    simplify = False,
    opset = 17
)

print("ONNX export completed")

device = 0 if torch.cuda.is_available() else "cpu"
print(device)

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=device,
    imgsz=640,
    conf=0.25,
    save=True
)

