from ultralytics import YOLO
import cv2
import os

IS_TRAIN = False
if IS_TRAIN:
   # Load the model.
   model = YOLO('yolov8n.pt')
   
   # Training.
   results = model.train(
      data='/home/openark/Desktop/yolo_bbox/yolo_bbox_dataset.yaml',
      imgsz=[1242, 2208],
      epochs=500,
      batch=8,
      name='yolov8n_v8_500e'
   )
else:
   # test
   model = YOLO("/home/openark/Desktop/yolo_bbox/runs/detect/yolov8n_v8_500e2/weights/best.pt")
   # result = model(['/home/openark/Desktop/yolo_bbox/test/0210.png'])[0]  # return a list of Results objects
   # boxes = result.boxes
   # print(boxes)
   model.predict(source = '/home/openark/Desktop/yolo_bbox/test', show = True, save=True)
   cv2.waitKey(1)