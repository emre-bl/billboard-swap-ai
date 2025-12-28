"""
Master Training Script.

Trains:
1. YOLOv8 Detection (n, s, m, l, x)
2. YOLOv8 Segmentation (n, s, m, l, x)
3. Mask R-CNN
"""
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from scripts.train_yolo_det import train_all_variants as train_det
from scripts.train_yolo_seg import train_all_variants as train_seg
from scripts.train_maskrcnn import train_maskrcnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--skip-det", action="store_true")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-maskrcnn", action="store_true")
    args = parser.parse_args()
    
    # 1. Train YOLO Detection
    if not args.skip_det:
        print("\n=== Starting YOLO Detection Training (All Variants) ===")
        train_det(data_yaml="data_detection.yaml", epochs=args.epochs)
    
    # 2. Train YOLO Segmentation
    if not args.skip_seg:
        print("\n=== Starting YOLO Segmentation Training (All Variants) ===")
        train_seg(data_yaml="data_segmentation.yaml", epochs=args.epochs)
        
    # 3. Train Mask R-CNN
    if not args.skip_maskrcnn:
        print("\n=== Starting Mask R-CNN Training ===")
        train_maskrcnn(
            # train_images argument at line 42 was a typo
            train_images=".", # Root for relative paths in JSON
            train_annotations="train_coco.json",
            val_images=".", 
            val_annotations="val_coco.json",
            epochs=args.epochs,
            batch_size=4 # Conservative
        )

    # 4. Run Benchmarks
    print("\n=== Running Benchmarks ===")
    import subprocess
    subprocess.run([sys.executable, "src/scripts/benchmark_all.py"])

if __name__ == "__main__":
    main()
