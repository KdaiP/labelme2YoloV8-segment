from ultralytics import YOLO
from ultralytics import settings

settings.update({'datasets_dir': './'})
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model
    results = model.train(data='./datasets/yolo.yaml', epochs=100, imgsz=640)