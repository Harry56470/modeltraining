from ultralytics import YOLO
model=YOLO("yolov8n.pt")
results=model.predict("goal.jpeg",save=True)
#model.predict(source="1",show=True)
for result in results:
    boxes = result.boxes  # Get the bounding boxes
    for box in boxes:
        # Get the class ID (e.g., 0 for person, 5 for bus)
        class_id = int(box.cls[0])
        # Get the name of the class
        label = model.names[class_id]
        # Get the confidence score (0.0 to 1.0)
        conf = float(box.conf[0])
        
        print(f"Detected {label} with {conf:.2f} confidence")