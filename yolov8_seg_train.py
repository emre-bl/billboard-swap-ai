from ultralytics import YOLO

def main():
    # 1. Load the pretrained YOLOv8-seg model
    model = YOLO("yolov8n-seg.pt") 

    # 2. Train the model
    # Ensure 'data.yaml' is in the same folder or provide the full path
    model.train(
        data="data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16,
        name="billboard_segmentation"
    )

if __name__ == '__main__':
    main()