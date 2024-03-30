from ultralytics import YOLO

# Load a model
model = YOLO("yolo-Weights/yolov8n.pt")  # load an official model


# Export the model
model.export(format="onnx")
