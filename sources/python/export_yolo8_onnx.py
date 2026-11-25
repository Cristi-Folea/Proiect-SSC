import torch 
import onnx 

from ultralytics import YOLO
from pathlib import Path

models_dir = Path(__file__).parent.parent / "models"

model = YOLO("yolov8n.pt")

exported = Path(model.export(
    format = "onnx",
    imgsz = 640,
    batch = 1,
    dynamic = False,
    simplify = False,
    opset = 17,
    
))

destination = models_dir / "yolov8n_fp32.onnx"
exported.rename(destination)

exported = Path(model.export(
    format = "onnx",
    imgsz = 640,
    batch = 1,
    dynamic = False,
    simplify = False,
    opset = 17,
    quantize = 16,
    ))

destination = models_dir / "yolov8n_fp16.onnx"
exported.rename(destination)

print("ONNX export completed")