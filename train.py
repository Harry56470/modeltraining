from ultralytics import YOLO
model=YOLO("yolov8n.pt")
model.train(data="Soccer-Ball-2/data.yaml",epochs=2)