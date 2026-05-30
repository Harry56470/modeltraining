import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import numpy
model=YOLO("models/yolov8n.pt")
def analyze_video(video_path):
    #setup videowritter for output video
    stream=cv2.VideoCapture(video_path)
    output_path="output"
    os.makedirs(output_path,exist_ok=True)
    output_video=Path(output_path)/f"{Path(video_path).stem}.mp4"
    fps=30
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out=cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height))

    #analyze each frame of video
    while True:
        ret,frame=stream.read()
        if ret:
            results=model(frame)
            detections=[]
            for result in results[0]:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                names = result.names

                for (x1, y1, x2, y2, conf, cls) in zip(*[boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confs, classes]):
                    name=names[cls]
                
                    detections.append((x1,y1,x2,y2,conf,name))