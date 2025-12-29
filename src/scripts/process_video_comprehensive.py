"""
Comprehensive Billboard Replacement Script.
Process every frame (Detect + Segment) without tracking optimization.
Supports all implemented approaches.

Modes:
- maskrcnn: Mask R-CNN (End-to-End)
- yolo_sam: YOLO Detection + SAM2 Segmentation (Box Prompt)
- yolo_seg: YOLO Segmentation (End-to-End)
- grounded_sam: GroundingDINO + SAM2 (Zero-Shot)
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.detection import create_detector, YOLODetector, GroundingDINODetector
from modules.segmentation import create_segmenter, MaskRCNNSegmenter, YOLOSegmenter, SAM2Segmenter, GroundedSAMSegmenter
from modules.replacement import apply_warp, CornerSmoother

def get_best_box(boxes, method="area"):
    if not boxes:
        return None
    if method == "area":
        # boxes in xyxy
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        return boxes[np.argmax(areas)]
    return boxes[0]

def process_video_mode(video_path, replacement_path, output_path, mode, config):
    print(f"\n--- Processing Mode: {mode} ---")
    
    # Initialize Models
    detector = None
    segmenter = None
    
    # Model Paths (Defaults)
    # All models are now in models/ directory
    yolo_det_path = "models/yolov8m_det_best.pt"
    yolo_seg_path = "models/yolov8m_seg2_best.pt"
    maskrcnn_path = "maskrcnn_trained_models/best.pth"
    sam2_path = "models/sam2_b.pt"
    grounding_path = "models/yolov8s-world.pt"
    
    try:
        if mode == "maskrcnn":
            print(f"Loading Mask R-CNN from {maskrcnn_path}...")
            if not Path(maskrcnn_path).exists():
                raise FileNotFoundError(f"Mask R-CNN model not found at {maskrcnn_path}")
            segmenter = MaskRCNNSegmenter(maskrcnn_path)
            
        elif mode == "yolo_sam":
             print(f"Loading YOLO Detector ({yolo_det_path}) and SAM2 ({sam2_path})...")
             if not Path(yolo_det_path).exists():
                  print(f"Warning: YOLO Det path not found {yolo_det_path}, using yolov8m.pt")
                  yolo_det_path = "yolov8m.pt"
             detector = YOLODetector(yolo_det_path)
             segmenter = SAM2Segmenter(sam2_path)
             
        elif mode == "yolo_seg":
            print(f"Loading YOLO Segmenter from {yolo_seg_path}...")
            if not Path(yolo_seg_path).exists():
                print(f"Warning: YOLO Seg path not found {yolo_seg_path}, using yolov8n-seg.pt")
                segmenter = YOLOSegmenter(variant="n") # Fallback
            else:
                segmenter = YOLOSegmenter(model_path=yolo_seg_path)
                
        elif mode == "grounded_sam":
            print("Loading Grounded SAM (Zero-Shot)...")
            segmenter = GroundedSAMSegmenter(
                sam_model_path=sam2_path,
                grounding_model_path=grounding_path,
                text_prompt="billboard"
            )
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        print(f"Failed to initialize models for {mode}: {e}")
        return

    # Video Setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Setup
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Replacement Image
    rep_img = cv2.imread(replacement_path)
    if rep_img is None:
        print(f"Error: Could not load replacement image {replacement_path}")
        cap.release()
        return

    # Smoother (Visual smoothing only, no tracking logic)
    smoother = CornerSmoother(alpha=0.6)
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        mask = None
        
        # --- PER-FRAME DETECTION & SEGMENTATION ---
        
        if mode == "maskrcnn":
            # End-to-end
            masks = segmenter.segment(frame, conf=0.5)
            mask = segmenter.select_best(masks, method="area")
            
        elif mode == "yolo_seg":
            # End-to-end
            masks = segmenter.segment(frame, conf=0.5)
            mask = segmenter.select_best(masks, method="area")
            
        elif mode == "yolo_sam":
            # Detect -> Segment
            boxes = detector.detect(frame, conf=0.5)
            best_box = get_best_box(boxes)
            if best_box:
                mask = segmenter.segment_with_box(frame, best_box)
                
        elif mode == "grounded_sam":
            # Zero-shot End-to-end (internal detect->segment)
            masks = segmenter.segment(frame, conf=0.3)
            mask = segmenter.select_best(masks, method="area")
            
        # --- REPLACEMENT ---
        result_frame = frame
        if mask is not None:
             # Ensure mask is correct size
             if mask.shape[:2] != (h, w):
                 mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                 
             try:
                 result_frame = apply_warp(frame, mask, rep_img, smoother=smoother)
             except Exception as e:
                 pass # Skip bad warps

        out.write(result_frame)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"\rProcessing: {frame_idx}/{total_frames} ({mode})", end="", flush=True)

    cap.release()
    out.release()
    print(f"\nSaved to {output_path}")
    
    # Cleanup to save VRAM
    del detector
    del segmenter
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Billboard Replacement (Per-Frame)")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--replacement", required=True, help="Replacement image path")
    parser.add_argument("--output-dir", default="outputs_comprehensive", help="Output directory")
    parser.add_argument("--mode", default="all", choices=["all", "maskrcnn", "yolo_sam", "yolo_seg", "grounded_sam"])
    
    args = parser.parse_args()
    
    modes = []
    if args.mode == "all":
        modes = ["maskrcnn", "yolo_sam", "yolo_seg", "grounded_sam"]
    else:
        modes = [args.mode]
        
    for mode in modes:
        output_filename = f"{Path(args.video).stem}_{mode}.mp4"
        output_path = str(Path(args.output_dir) / output_filename)
        
        process_video_mode(args.video, args.replacement, output_path, mode, args)

if __name__ == "__main__":
    main()
