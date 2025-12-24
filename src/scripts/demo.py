import sys
import os
import cv2
import argparse
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.detection import BillboardDetector, HybridDetector, MaskRCNNDetector
from modules.segmentation import GroundedSAM
from modules.replacement import apply_warp, CornerSmoother
from modules.tracking import OpticalFlowTracker

def main():
    parser = argparse.ArgumentParser(description="Billboard Replacement Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--image", required=True, help="Path to replacement image")
    parser.add_argument("--output", default="output.mp4", help="Path to output video")
    parser.add_argument("--model-type", choices=["fine-tuned", "zero-shot", "hybrid", "mask-rcnn"], default="fine-tuned", help="Model type")
    parser.add_argument("--model-path", default="runs/segment/billboard_segmentation5/weights/best.pt", help="Path to Fine-Tuned model weights")
    parser.add_argument("--track", action="store_true", help="Enable tracking (Optical Flow)")
    parser.add_argument("--detect-interval", type=int, default=1, help="Run detection every N frames (default 1 = every frame). Use 5-10 for speed.")
    
    args = parser.parse_args()
    
    # ... (Load Image and Video code unchanged) ...
    # Load Image
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
        
    repl_img = Image.open(args.image)
    repl_img = cv2.cvtColor(np.array(repl_img), cv2.COLOR_RGB2BGR)
    
    # Load Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize Model
    detector = None
    if args.model_type == "fine-tuned":
        print(f"Loading Fine-Tuned Model ({args.model_path})...")
        if not os.path.exists(args.model_path):
             print(f"Warning: Model not found at {args.model_path}")
        detector = BillboardDetector(model_path=args.model_path)
        
    elif args.model_type == "hybrid":
        print(f"Loading Hybrid Model (YOLO: {args.model_path} + SAM2)...")
        detector = HybridDetector(yolo_path=args.model_path, sam_path="sam2_b.pt")

    elif args.model_type == "mask-rcnn":
        print(f"Loading Mask R-CNN Model ({args.model_path})...")
        if not os.path.exists(args.model_path):
             print(f"Warning: Model not found at {args.model_path}")
        detector = MaskRCNNDetector(model_path=args.model_path)
        
    else:
        print("Loading Zero-Shot Model (GroundedSAM)...")
        detector = GroundedSAM(sam_model_path="sam2_b.pt", grounding_model_path="yolov8s-world.pt")
        
    tracker = None
    if args.track:
        print("Initializing Optical Flow Tracker...")
        tracker = OpticalFlowTracker()
        
    smoother = CornerSmoother(alpha=0.2) # Alpha 0.2 means slow update, very smooth
    
    mask = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_mask = None
        
        # Tracking logic
        if tracker and mask is not None:
            tracked_mask = tracker.track(frame)
            if tracked_mask is not None:
                current_mask = tracked_mask
        
        # Detection logic
        # Run if: (a) No mask found yet (tracking failed/lost) OR (b) Keyframe interval hit
        should_detect = (current_mask is None) or (args.detect_interval > 0 and frame_count % args.detect_interval == 0)
        
        if should_detect:
            masks = []
            if args.model_type == "fine-tuned":
                masks = detector.segment(frame)
            elif args.model_type == "hybrid":
                 masks = detector.detect_and_segment(frame)
            elif args.model_type == "mask-rcnn":
                 masks = detector.segment(frame)
            else:
                masks, _ = detector.detect_and_segment(frame, text_prompt="billboard")
            
            if len(masks) > 0:
                detected_mask = (masks[0] * 255).astype(np.uint8) if args.model_type in ["fine-tuned", "mask-rcnn"] else masks[0].astype(np.uint8) * 255
                
                # Resize if needed
                if mask is not None and mask.shape != detected_mask.shape:
                        detected_mask = cv2.resize(detected_mask, (width, height))
                
                current_mask = detected_mask
                
                # Re-initialize tracker with new high-quality mask
                if tracker:
                    tracker.init(frame, current_mask)
        
        mask = current_mask
        
        # Apply replacement
        if mask is not None:
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
                
            frame = apply_warp(frame, mask, repl_img, smoother=smoother)
            
        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
