"""
YOLOv8 Detection Training Script.

Trains YOLOv8 variants (n, s, m, l, x) for billboard detection.
"""
import argparse
from ultralytics import YOLO

VARIANTS = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt", 
    "m": "yolov8m.pt",
}

def train_yolo_det(
    variant: str = "n",
    data_yaml: str = "data_detection.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 15,
    project: str = "runs/detect",
    name: str = None,
):
    """Train YOLOv8 detection model."""
    
    if variant not in VARIANTS:
        raise ValueError(f"Invalid variant: {variant}")
    
    model_name = VARIANTS[variant]
    run_name = name or f"billboard_yolov8{variant}_det"
    
    print(f"Training {model_name} on {data_yaml}")
    
    model = YOLO(model_name)
    
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        project=project,
        name=run_name,
        augment=True,
        save=True,
    )
    
    print(f"Training complete: {project}/{run_name}")

def train_all_variants(data_yaml, epochs):
    """Train all variants."""
    for variant in VARIANTS.keys():
        try:
            train_yolo_det(variant, data_yaml, epochs)
        except Exception as e:
            print(f"Error training {variant}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="n", choices=list(VARIANTS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data", default="data_detection.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if args.all:
        train_all_variants(args.data, args.epochs)
    else:
        train_yolo_det(args.variant, args.data, args.epochs)

if __name__ == "__main__":
    main()
