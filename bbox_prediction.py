import math

import cv2
from ultralytics import YOLO

# model
model = YOLO("yolo-Weights/yolov8n.pt")


def estimate_depth(bbox, actual_width, focal_length):
    # Depth estimation function as defined previously
    bbox_width = bbox[2] - bbox[0]
    scale_factor = actual_width / bbox_width
    depth = focal_length * scale_factor
    return depth


def bbox_prediction(img):
    results = model(img, stream=True)

    # coordinatesc
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, str(confidence), org, font, fontScale, color, thickness)

            bbox = (x1, y1, x2, y2)
            depth = estimate_depth(bbox, 0.447, 1066)
            # （bbox，width of rover in m，focal length in pixel）
            print("Depth --->", depth)

    cv2.imshow("Webcam", img)
