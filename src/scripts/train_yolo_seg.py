"""
YOLOv8-seg Training Script for Billboard Segmentation.

Supports training all YOLO variants: n, s, m, l, x
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


VARIANTS = {
    "n": "yolov8n-seg.pt",
    "s": "yolov8s-seg.pt", 
    "m": "yolov8m-seg.pt",
}


def train_yolo_seg(
    variant: str = "n",
    data_yaml: str = "data_segmentation.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 10,
    workers: int = 4,
    device: str = None,
    resume: bool = False,
    project: str = "runs/segment",
    name: str = None,
):
    """Train YOLOv8-seg model for billboard segmentation."""
    
    if variant not in VARIANTS:
        raise ValueError(f"Invalid variant: {variant}. Choose from: {list(VARIANTS.keys())}")
    
    # Model name
    model_name = VARIANTS[variant]
    run_name = name or f"billboard_yolov8{variant}_seg"
    
    print(f"="*60)
    print(f"Training YOLOv8{variant}-seg")
    print(f"="*60)
    print(f"  Model: {model_name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Output: {project}/{run_name}")
    print(f"="*60)
    
    # Load pretrained model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        workers=workers,
        device=device,
        resume=resume,
        project=project,
        name=run_name,
        # Augmentation
        augment=True,
        # Save best model
        save=True,
        save_period=-1,  # Save only best
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {project}/{run_name}/weights/best.pt")
    
    return results


def train_all_variants(
    data_yaml: str = "data_merged.yaml",
    epochs: int = 50,
    batch_size: int = 16,
):
    """Train all YOLO variants sequentially."""
    
    results = {}
    
    for variant in VARIANTS.keys():
        print(f"\n{'='*60}")
        print(f"Training variant: {variant}")
        print(f"{'='*60}\n")
        
        try:
            result = train_yolo_seg(
                variant=variant,
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
            )
            results[variant] = {"status": "success", "result": result}
        except Exception as e:
            results[variant] = {"status": "error", "error": str(e)}
            print(f"Error training {variant}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for variant, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} yolov8{variant}-seg: {result['status']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for Billboard Segmentation")
    
    parser.add_argument("--variant", type=str, default="n", 
                        choices=list(VARIANTS.keys()),
                        help="YOLO variant: n, s, m, l, x")
    parser.add_argument("--all", action="store_true",
                        help="Train all variants sequentially")
    parser.add_argument("--data", type=str, default="data_merged.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., '0' for GPU 0, 'cpu' for CPU)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom run name")
    
    args = parser.parse_args()
    
    if args.all:
        train_all_variants(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
        )
    else:
        train_yolo_seg(
            variant=args.variant,
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            workers=args.workers,
            device=args.device,
            resume=args.resume,
            name=args.name,
        )


if __name__ == "__main__":
    main()
