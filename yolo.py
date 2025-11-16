from ultralytics import YOLO

# Load a pre-trained YOLOv8 classification model (using nano size for less memory)
model = YOLO("yolov8n-cls.pt")  # using nano model instead of small

# Train the classification model
model.train(
    data="/home/mry/forest_fire_detection/dataset/images",  # Root directory containing train and val folders
    epochs=50,
    imgsz=224,  # smaller image size
    batch=8,    # smaller batch size
    project="fire_detection_yolo",
    name="yolo_fire_v1_cls",
    device=0    # GPU
)
