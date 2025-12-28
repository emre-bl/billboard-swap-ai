"""
Video Processing Script for Billboard Replacement.
Supports: YOLOv8-seg, GroundedSAM (zero-shot), SAM2 (zero-shot)
Now supports Tracking (Detect once + Track) using SAM2, Optical Flow, or Cutie.
"""
import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.segmentation import YOLOSegmenter, GroundedSAMSegmenter, SAM2Segmenter
from modules.tracking import create_tracker, BaseTracker
from modules.replacement import apply_warp, CornerSmoother

class Config:
    """Simple config class to mimic PipelineConfig for factories."""
    def __init__(self, tracker="sam2", sam2_model_path="sam2_b.pt"):
        self.tracker = tracker
        self.sam2_model_path = sam2_model_path

def process_with_tracking(video_path, replacement_path, output_path, config):
    """
    Process video using Detect + Track approach.
    1. Detect billboard in 1st frame (or keyframes).
    2. Track mask in subsequent frames.
    """
    print(f"Processing with Tracking: {config.tracker}")
    
    # Initialize components
    # 1. Segmenter (Detection)
    if config.model_type == "yolo":
        print(f"Loading YOLOv8-seg: {config.model_path}")
        segmenter = YOLOSegmenter(model_path=config.model_path)
    elif config.model_type == "grounded-sam":
        print("Loading GroundedSAM...")
        segmenter = GroundedSAMSegmenter(text_prompt=config.prompt)
    elif config.model_type == "sam2":
        print("Loading SAM2 Segmenter...")
        segmenter = SAM2Segmenter()
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    # 2. Tracker
    tracker = create_tracker(config)
    
    # 3. Smoother & Replacement
    smoother = CornerSmoother(alpha=0.7) # Smoothing for video consistency
    replacement_img = cv2.imread(replacement_path)
    if replacement_img is None:
        raise ValueError(f"Could not load replacement image: {replacement_path}")
        
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    detections = 0
    tracking_failures = 0
    current_mask = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        result_frame = frame.copy()
        mask = None
        
        # LOGIC:
        # Frame 0: Detect
        # Frame N: Track
        # If Track Fails: Re-detect
        
        if frame_idx == 0 or current_mask is None:
            # Detection Phase
            masks = segmenter.segment(frame, conf=config.confidence)
            current_mask = segmenter.select_best(masks, method="area")
            
            if current_mask is not None:
                # Initialize Tracker
                tracker.init(frame, current_mask)
                mask = current_mask
                detections += 1
            else:
                # No detection found
                pass
        else:
            # Tracking Phase
            mask = tracker.track(frame)
            
            if mask is None:
                # Tracking Lost -> Try Re-detect
                tracking_failures += 1
                masks = segmenter.segment(frame, conf=config.confidence)
                mask = segmenter.select_best(masks, method="area")
                
                if mask is not None:
                    tracker.init(frame, mask) # Re-init tracker
                    detections += 1
            else:
                current_mask = mask # Update current mask for ref
        
        # Apply Replacement if mask exists
        if mask is not None:
             # Ensure mask is binary and right size
            if mask.shape[:2] != (height, width):
                 mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Apply
            try:
                result_frame = apply_warp(frame, mask, replacement_img, smoother=smoother)
            except Exception as e:
                pass # Skip failed frames
                
        out.write(result_frame)
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            status = "Detect" if tracking_failures > 0 else "Track"
            print(f"\r[{status}] Progress: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)", end="", flush=True)
            
    cap.release()
    out.release()
    print(f"\nProcessing Complete: {frame_idx} frames. Tracking Failures/Redetections: {tracking_failures}")


# --- Legacy / Direct Processing Functions (kept for backward compatibility if needed) ---
# These are largely superseded by process_with_tracking but kept if user wants pure frame-by-frame
def process_frame_by_frame(video_path, replacement_path, output_path, model_type="yolo", model_path="runs/segment/billboard_yolov8m_seg/weights/best.pt", confidence=0.55):
    """Old method: Detect every frame independently."""
    # We can reuse the process_with_tracking logic but force re-detection every frame if we really wanted,
    # but for now we'll just redirect to the tracking one with a dummy tracker or just suggest using tracking.
    print("Warning: Running in frame-by-frame mode (Slow). Use --tracking for speed.")
    
    cfg = Config()
    cfg.model_type = model_type
    cfg.model_path = model_path
    cfg.confidence = confidence
    cfg.prompt = "billboard"
    cfg.tracker = "optical-flow" # Dummy, won't be used if we force re-detect? 
    # Actually, let's just use the new pipeline. It's better.
    process_with_tracking(video_path, replacement_path, output_path, cfg)


def main():
    parser = argparse.ArgumentParser(description="Billboard Replacement Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--replacement", required=True, help="Replacement image path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--model-type", choices=["yolo", "grounded-sam", "sam2"], default="yolo",
                        help="Segmentation model type (used for initial detection)")
    parser.add_argument("--model-path", default="runs/segment/billboard_yolov8m_seg/weights/best.pt", 
                        help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--prompt", default="billboard", help="Text prompt for GroundedSAM")
    
    # Tracking args
    parser.add_argument("--tracking", action="store_true", help="Enable tracking (Detect -> Track)")
    parser.add_argument("--tracker", choices=["sam2", "optical-flow", "cutie"], default="sam2",
                        help="Tracker type to use")
    
    args = parser.parse_args()
    
    print(f"Input: {args.video}")
    print(f"Replacement: {args.replacement}")
    print(f"Model: {args.model_type}")
    
    # Create Config
    cfg = Config(tracker=args.tracker)
    cfg.model_type = args.model_type
    cfg.model_path = args.model_path
    cfg.confidence = args.confidence
    cfg.prompt = args.prompt
    
    # Default to tracking if not specified, or if specified
    # The user request implies we SHOULD use tracking now.
    # So we used the shared pipeline.
    
    process_with_tracking(args.video, args.replacement, args.output, cfg)


if __name__ == "__main__":
    main()
