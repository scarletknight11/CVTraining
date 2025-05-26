from ultralytics import YOLO
import os

# Suppress OpenMP duplicate runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load your trained YOLO model
model = YOLO("yolo11m-seg-custom.pt")

# Export only to TFLite without trying to install extras
model.export(format="tflite", nms=False, optimize=False, int8=False)
