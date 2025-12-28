"""
Billboard Replacement Pipeline V2.

Implements the Detect -> Segment -> Track workflow.
"""
import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.detection import create_detector
from modules.segmentation import create_segmenter, SAM2Segmenter
from modules.tracking import create_tracker
from modules.replacement import apply_warp, CornerSmoother

class PipelineConfig:
    def __init__(self, args):
        self.det_model_type = args.det_model
        self.det_path = args.det_path
        self.seg_model_type = args.seg_model
        self.seg_path = args.seg_path
        self.tracker_type = args.tracker
        self.confidence = args.conf
        self.prompt = args.prompt
        
        # Mapping for factory compatibility
        self.tracker = args.tracker 
        self.sam2_model_path = "sam2_b.pt"

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_video(args):
    """Run the pipeline."""
    config = PipelineConfig(args)
    
    print(f"Pipeline: Detect({config.det_model_type}) -> Segment({config.seg_model_type}) -> Track({config.tracker_type})")
    
    # 1. Initialize Detector
    detector = create_detector(config.det_model_type, config.det_path, config.prompt)
    
    # 2. Initialize Segmenter
    # For SAM/SAM2, we instantiate directly to use box prompting
    if config.seg_model_type in ["sam", "sam2"]:
        segmenter = SAM2Segmenter() # or original SAM
        is_promptable = True
    else:
        # YOLO-seg, Mask R-CNN, etc.
        # We need to map args to create_segmenter config expectation
        # Quick hack: create a dummy config object with expected fields
        class SegConfig:
            model_type = config.seg_model_type
            model_path = config.seg_path
            text_prompt = config.prompt
            
        segmenter = create_segmenter(SegConfig())
        is_promptable = False
        
    # 3. Initialize Tracker
    tracker = create_tracker(config)
    
    # 4. Replacement Load
    replacement_img = cv2.imread(args.replacement)
    if replacement_img is None:
        raise ValueError(f"Could not load replacement: {args.replacement}")
    
    # Video Setup
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    smoother = CornerSmoother(alpha=0.6)
    
    tracking_active = False
    
    frame_idx = 0
    detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        mask = None
        
        # --- Logic ---
        
        if not tracking_active:
            # 1. DETECT
            boxes = detector.detect(frame, conf=config.confidence)
            best_box = detector.select_best(boxes, method="area")
            
            if best_box is not None:
                # 2. SEGMENT
                if is_promptable:
                    # SAM: Prompt with Box
                    mask = segmenter.segment_with_box(frame, best_box)
                else:
                    # Traditional: Segment entire image, find mask matching box
                    masks = segmenter.segment(frame, conf=config.confidence)
                    # Find mask with best overlap with best_box
                    best_iou = 0
                    best_mask = None
                    for m in masks:
                        # Mask to box
                        ys, xs = np.where(m > 0)
                        if len(xs) > 0:
                            mbox = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
                            iou = compute_iou(best_box, mbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_mask = m
                    
                    if best_iou > 0.5: # Threshold for match
                        mask = best_mask
                
                # 3. INIT TRACKER
                if mask is not None:
                    tracker.init(frame, mask)
                    tracking_active = True
                    detections += 1
        
        else:
            # TRACK
            mask = tracker.track(frame)
            if mask is None:
                tracking_active = False # Lost tracking, go back to detect next frame
        
        # Apply
        result_frame = frame
        if mask is not None:
             if mask.shape[:2] != (h, w):
                 mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
             
             try:
                 result_frame = apply_warp(frame, mask, replacement_img, smoother=smoother)
             except:
                 pass
        
        out.write(result_frame)
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            status = "Track" if tracking_active else "Scan"
            print(f"\r[{status}] {frame_idx}/{total_frames}", end="", flush=True)

    cap.release()
    out.release()
    print("\nDone.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--output", default="output_v2.mp4")
    
    # Detection
    parser.add_argument("--det-model", default="yolo", choices=["yolo", "grounding-dino"])
    parser.add_argument("--det-path", default="yolov8n.pt")
    
    # Segmentation
    parser.add_argument("--seg-model", default="sam2", choices=["sam2", "yolo", "maskrcnn"])
    parser.add_argument("--seg-path", default="sam2_b.pt")
    
    # Tracking
    parser.add_argument("--tracker", default="sam2", choices=["sam2", "optical-flow", "cutie"])
    
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--prompt", default="billboard")
    
    args = parser.parse_args()
    process_video(args)

if __name__ == "__main__":
    main()
