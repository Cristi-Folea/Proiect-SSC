#Numele etichetelor:

from ultralytics import YOLO
model = YOLO('yolov8n.pt');

with open("etichete.txt", "w") as f:
         for key, value in model.names.items():
             f.write(value + "\n")