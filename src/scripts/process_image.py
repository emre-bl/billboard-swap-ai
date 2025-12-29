"""
Billboard Replacement for Single Image.
Uses various segmentation models including YOLOv8-seg, GroundedSAM, SAM2.
"""
import sys
import os
import cv2
import argparse
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.segmentation import YOLOSegmenter, GroundedSAMSegmenter, SAM2Segmenter
from modules.replacement import apply_warp, CornerSmoother


def main():
    parser = argparse.ArgumentParser(description="Billboard Replacement for Single Image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--replacement", required=True, help="Path to replacement image")
    parser.add_argument("--output", default="output.png", help="Path to output image")
    parser.add_argument("--model-type", choices=["yolo", "grounded-sam", "sam2"], default="yolo",
                        help="Segmentation model type")
    parser.add_argument("--model-path", default=None, help="Path to model weights (for YOLO)")
    parser.add_argument("--yolo-variant", default="m", choices=["n", "s", "m", "l", "x"],
                        help="YOLO variant")
    parser.add_argument("--prompt", default="billboard", help="Text prompt for zero-shot models")
    parser.add_argument("--conf", type=float, default=0.55, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Load Input Image
    if not os.path.exists(args.input):
        print(f"Input image not found: {args.input}")
        return
        
    input_img = cv2.imread(args.input)
    if input_img is None:
        print(f"Failed to read input image: {args.input}")
        return
    
    height, width = input_img.shape[:2]
    print(f"Input image size: {width}x{height}")
    
    # Load Replacement Image
    if not os.path.exists(args.replacement):
        print(f"Replacement image not found: {args.replacement}")
        return
        
    repl_img = Image.open(args.replacement)
    repl_img = cv2.cvtColor(np.array(repl_img), cv2.COLOR_RGB2BGR)
    print(f"Replacement image size: {repl_img.shape[1]}x{repl_img.shape[0]}")

    # Initialize Segmenter
    segmenter = None
    if args.model_type == "yolo":
        model_path = args.model_path or f"models/yolov8{args.yolo_variant}_seg_best.pt"
        print(f"Loading YOLOv8{args.yolo_variant}-seg ({model_path})...")
        segmenter = YOLOSegmenter(model_path=model_path, variant=args.yolo_variant)
        
    elif args.model_type == "grounded-sam":
        print("Loading GroundedSAM (zero-shot)...")
        segmenter = GroundedSAMSegmenter(text_prompt=args.prompt)
        
    elif args.model_type == "sam2":
        print("Loading SAM2 (zero-shot)...")
        segmenter = SAM2Segmenter()
    
    smoother = CornerSmoother(alpha=1.0)  # No smoothing for single image
    
    # Run Segmentation
    print("Running segmentation...")
    masks = segmenter.segment(input_img, conf=args.conf)
    
    print(f"Found {len(masks)} billboard(s)")
    
    if len(masks) == 0:
        print("No billboards detected! Saving original image.")
        cv2.imwrite(args.output, input_img)
        return
    
    # Select best mask
    mask = segmenter.select_best(masks, method="area")
    
    # Ensure binary format
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Resize mask if needed
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height))
    
    # Apply replacement
    print("Applying replacement...")
    output_img = apply_warp(input_img, mask, repl_img, smoother=smoother)
    
    # Save output
    cv2.imwrite(args.output, output_img)
    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
