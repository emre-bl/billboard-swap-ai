"""
Billboard Replacement Pipeline V3 - Enhanced Tracking.

Implements the Detect -> Segment -> Track workflow with improved tracking.
"""
import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.detection import create_detector
from modules.segmentation import create_segmenter, SAM2Segmenter
from modules.tracking import create_tracker, create_fusion_tracker
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
        self.fusion = args.fusion
        
        # Mapping for factory compatibility
        self.tracker = args.tracker 
        self.sam2_model_path = args.seg_path if ("sam2" in args.seg_path.lower() and args.seg_path.endswith('.pt')) else "models/sam2_b.pt"
        self.adaptive_tracking = args.adaptive
        self.motion_history = args.motion_history

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    if box1 is None or box2 is None:
        return 0.0
        
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

def is_mask_sane(mask, prev_area=None, prev_center=None, motion_threshold=100):
    """
    Enhanced mask sanity check with motion constraints.
    """
    if mask is None: 
        return False
    
    # Check if mask has any content
    if np.sum(mask > 127) < 100:
        return False
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return False
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    # 1. Area check
    h, w = mask.shape
    min_area_ratio = 0.0005  # At least 0.05% of frame area
    max_area_ratio = 0.5    # At most 50% of frame area
    frame_area = h * w
    
    if area < frame_area * min_area_ratio or area > frame_area * max_area_ratio:
        return False
    
    # 2. Area change rate (optional)
    if prev_area:
        area_ratio = area / prev_area if prev_area > 0 else 1.0
        if area_ratio > 2.0 or area_ratio < 0.5:
            return False

    # 3. Convexity check
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    if solidity < 0.7:  # Slightly more tolerant
        return False
    
    # 4. Aspect ratio check (billboards are typically rectangular)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    if aspect_ratio > 10:  # Too elongated
        return False
    
    # 5. Motion consistency check
    if prev_center is not None:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            motion = np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)
            if motion > motion_threshold:
                return False
    
    # 6. Circularity/compactness check
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.1:  # Too complex shape
            return False
    
    return True

def refine_mask(mask, kernel_size=3, iterations=1):
    """Post-process mask to remove noise and fill holes."""
    if mask is None:
        return None
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove small noise
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Fill holes
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Keep only the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_clean = np.uint8(labels == largest_label) * 255
    
    return mask_clean

def compute_mask_similarity(mask1, mask2):
    """Compute similarity between two masks using IoU."""
    if mask1 is None or mask2 is None:
        return 0.0
    
    mask1_bin = (mask1 > 127).astype(np.uint8)
    mask2_bin = (mask2 > 127).astype(np.uint8)
    
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    return intersection / union if union > 0 else 0.0

class AdaptiveTracker:
    """Wrapper for adaptive tracking with confidence estimation."""
    
    def __init__(self, config, base_tracker):
        self.tracker = base_tracker
        self.config = config
        self.tracking_confidence = 1.0
        self.consecutive_failures = 0
        self.mask_history = []
        self.max_history = 10
        self.last_valid_mask = None
        self.last_valid_frame = None
        
    def init(self, frame, mask):
        self.tracker.init(frame, mask)
        self.tracking_confidence = 1.0
        self.consecutive_failures = 0
        self.mask_history = [mask.copy()]
        self.last_valid_mask = mask.copy()
        self.last_valid_frame = frame.copy()
        
    def track(self, frame):
        # Try to track
        mask = self.tracker.track(frame)
        
        # If tracking failed, try recovery strategies
        if mask is None or not is_mask_sane(mask):
            self.consecutive_failures += 1
            
            # Strategy 1: Use motion prediction
            if self.last_valid_mask is not None and len(self.mask_history) >= 2:
                mask = self.predict_from_history(frame)
            
            # Strategy 2: Fall back to last valid mask
            if (mask is None or not is_mask_sane(mask)) and self.last_valid_mask is not None:
                mask = self.last_valid_mask.copy()
                print("[WARN] Using last valid mask as fallback")
            
            # Update confidence
            self.tracking_confidence = max(0.1, self.tracking_confidence * 0.7)
        else:
            # Tracking succeeded
            self.consecutive_failures = 0
            mask = refine_mask(mask)
            
            # Update history
            self.mask_history.append(mask.copy())
            if len(self.mask_history) > self.max_history:
                self.mask_history.pop(0)
            
            # Update confidence
            similarity = self.compute_temporal_consistency(mask)
            self.tracking_confidence = min(1.0, self.tracking_confidence * 0.9 + similarity * 0.1)
            
            self.last_valid_mask = mask.copy()
            self.last_valid_frame = frame.copy()
        
        return mask
    
    def predict_from_history(self, frame):
        """Predict mask position based on motion history."""
        if len(self.mask_history) < 2:
            return None
        
        # Simple linear motion prediction
        prev_mask = self.mask_history[-1]
        prev_prev_mask = self.mask_history[-2] if len(self.mask_history) >= 2 else None
        
        if prev_prev_mask is None:
            return prev_mask
        
        # Estimate motion using optical flow on mask corners
        try:
            # Get contours
            contours1, _ = cv2.findContours(prev_prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours1 and contours2:
                cnt1 = max(contours1, key=cv2.contourArea)
                cnt2 = max(contours2, key=cv2.contourArea)
                
                # Get moments
                M1 = cv2.moments(cnt1)
                M2 = cv2.moments(cnt2)
                
                if M1['m00'] > 0 and M2['m00'] > 0:
                    cx1 = int(M1['m10'] / M1['m00'])
                    cy1 = int(M1['m01'] / M1['m00'])
                    cx2 = int(M2['m10'] / M2['m00'])
                    cy2 = int(M2['m01'] / M2['m00'])
                    
                    # Estimate translation
                    dx = cx2 - cx1
                    dy = cy2 - cy1
                    
                    # Apply translation to last mask
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    h, w = prev_mask.shape
                    predicted_mask = cv2.warpAffine(prev_mask, M, (w, h))
                    
                    return predicted_mask
        except:
            pass
        
        return prev_mask
    
    def compute_temporal_consistency(self, current_mask):
        """Check consistency with recent history."""
        if len(self.mask_history) == 0:
            return 1.0
        
        similarities = []
        for hist_mask in self.mask_history[-3:]:  # Check last 3 frames
            sim = compute_mask_similarity(current_mask, hist_mask)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_confidence(self):
        return self.tracking_confidence

def process_video_enhanced(args):
    config = PipelineConfig(args)
    
    # SegConfig helper class
    class SegConfig:
        def __init__(self, c):
            self.segmenter = "yolov8-seg" if c.seg_model_type == "yolo" else \
                            "mask-rcnn" if c.seg_model_type == "maskrcnn" else \
                            c.seg_model_type
            self.yolo_model_path = c.seg_path
            self.yolo_variant = "n"
            self.maskrcnn_model_path = c.seg_path
            self.sam2_model_path = c.seg_path if "sam2" in c.seg_path else "sam2_b.pt"
            self.text_prompt = c.prompt
    
    # Initialize components
    detector = create_detector(config.det_model_type, config.det_path, config.prompt)
    seg_config = SegConfig(config)
    segmenter = create_segmenter(seg_config)
    is_promptable = hasattr(segmenter, 'segment_with_box')
    
    # Create tracker with optional fusion
    if config.fusion:
        tracker = create_fusion_tracker(config)
    else:
        base_tracker = create_tracker(config)
        if config.adaptive_tracking:
            tracker = AdaptiveTracker(config, base_tracker)
        else:
            tracker = base_tracker
    
    replacement_img = cv2.imread(args.replacement)
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    smoother = CornerSmoother(alpha=0.6)
    
    # Enhanced tracking parameters
    tracking_active = False
    frame_idx = 0
    last_valid_area = None
    last_valid_center = None
    detection_interval = args.detection_interval
    iou_threshold = args.iou_threshold
    
    # Performance metrics
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracking_time = 0
    detection_time = 0
    
    print(f"\nStarting enhanced tracking pipeline...")
    print(f"Video: {args.video}")
    print(f"Frames: {total_frames}, FPS: {fps}")
    print(f"Tracker: {config.tracker_type}{' (adaptive)' if config.adaptive_tracking else ''}")
    print(f"Detection interval: Every {detection_interval} frames")
    print("-" * 50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        current_mask = None
        
        # Calculate adaptive detection interval based on tracking confidence
        if hasattr(tracker, 'get_confidence'):
            tracker_confidence = tracker.get_confidence()
            adaptive_interval = max(5, min(60, int(detection_interval * tracker_confidence)))
            force_re_detect = (frame_idx % adaptive_interval == 0)
        else:
            force_re_detect = (frame_idx % detection_interval == 0)
        
        if not tracking_active or force_re_detect:
            # DETECTION PHASE
            start_time = time.time()
            boxes = detector.detect(frame, conf=config.confidence)
            detection_time += time.time() - start_time
            
            best_box = detector.select_best(boxes, method="area")
            
            if best_box is not None:
                if not tracking_active:
                    # New detection - initialize tracker
                    if is_promptable:
                        current_mask = segmenter.segment_with_box(frame, best_box)
                    else:
                        masks = segmenter.segment(frame)
                        current_mask = masks[0] if masks else None
                    
                    if current_mask is not None and is_mask_sane(current_mask):
                        tracker.init(frame, current_mask)
                        tracking_active = True
                        print(f"\n[INIT] Frame {frame_idx}: Billboard detected, tracker started.")
                else:
                    # Validation phase - compare tracker with detector
                    start_time = time.time()
                    tracked_mask = tracker.track(frame)
                    tracking_time += time.time() - start_time
                    
                    if tracked_mask is not None:
                        tracked_box = mask_to_bbox(tracked_mask)
                        iou = compute_iou(best_box, tracked_box)
                        
                        if iou < iou_threshold:
                            # Tracker has drifted - reinitialize with detection
                            if is_promptable:
                                current_mask = segmenter.segment_with_box(frame, best_box)
                            else:
                                masks = segmenter.segment(frame)
                                current_mask = masks[0] if masks else None
                            
                            if current_mask is not None:
                                tracker.init(frame, current_mask)
                                print(f"\n[CORRECT] Frame {frame_idx}: Drift detected (IoU: {iou:.2f}). Re-initializing.")
                        else:
                            current_mask = tracked_mask
                            # Update tracker with detection (optional - can help prevent drift)
                            if hasattr(tracker, 'update_measurement') and iou > 0.8:
                                tracker.update_measurement(current_mask)
            else:
                # No detection - rely on tracker
                if tracking_active:
                    start_time = time.time()
                    current_mask = tracker.track(frame)
                    tracking_time += time.time() - start_time
                    
                    if current_mask is None or not is_mask_sane(current_mask, last_valid_area, last_valid_center):
                        tracking_active = False
                        current_mask = None
        else:
            # TRACKING PHASE
            start_time = time.time()
            current_mask = tracker.track(frame)
            tracking_time += time.time() - start_time
            
            # Enhanced sanity check with motion consistency
            if not is_mask_sane(current_mask, last_valid_area, last_valid_center):
                tracking_active = False
                current_mask = None
                print(f"\n[LOST] Frame {frame_idx}: Mask failed sanity check.")
        
        # Apply replacement and record
        if current_mask is not None:
            # Update tracking metrics
            last_valid_area = np.sum(current_mask > 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    last_valid_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            try:
                result_frame = apply_warp(frame, current_mask, replacement_img, smoother=smoother)
            except Exception as e:
                print(f"\n[ERROR] Frame {frame_idx}: Warp failed - {e}")
                result_frame = frame
        else:
            result_frame = frame
            last_valid_area = None
            last_valid_center = None
        
        out.write(result_frame)
        frame_idx += 1
        
        # Progress reporting
        if frame_idx % 30 == 0:
            status = "TRACKING" if tracking_active else "SCANNING"
            progress = (frame_idx / total_frames) * 100
            conf = tracker.get_confidence() if hasattr(tracker, 'get_confidence') else 1.0
            print(f"\r[{status}] Frame: {frame_idx}/{total_frames} ({progress:.1f}%) | Confidence: {conf:.2f}", end="")
    
    cap.release()
    out.release()
    
    # Performance summary
    print(f"\n\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total tracking time: {tracking_time:.2f}s")
    print(f"Total detection time: {detection_time:.2f}s")
    print(f"Average FPS: {frame_idx / (tracking_time + detection_time):.1f}")
    print(f"Tracking active for: {frame_idx - sum(1 for i in range(frame_idx) if i % detection_interval == 0)} frames")
    print(f"Output saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Billboard Replacement Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--replacement", required=True, help="Replacement image path")
    parser.add_argument("--output", default="output_enhanced.mp4", help="Output video path")
    
    # Detection
    parser.add_argument("--det-model", default="yolo", 
                       choices=["yolo", "none"])
    parser.add_argument("--det-path", default="yolov8n.pt")
    
    # Segmentation
    parser.add_argument("--seg-model", default="sam2", 
                       choices=["sam2", "yolo", "maskrcnn"])
    parser.add_argument("--seg-path", default="sam2_b.pt")
    
    # Tracking
    parser.add_argument("--tracker", default="planar-kalman",
                       choices=["sam2", "optical-flow", "kalman", "bytetrack", 
                               "feature-homography", "ecc-homography", 
                               "planar-kalman", "sam2-memory", "hybrid-flow"])
    parser.add_argument("--fusion", action="store_true", 
                       help="Use fusion of multiple trackers")
    parser.add_argument("--adaptive", action="store_true", 
                       help="Use adaptive tracking with confidence estimation")
    parser.add_argument("--motion-history", type=int, default=10,
                       help="Motion history buffer size")
    
    # Parameters
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="Detection confidence threshold")
    parser.add_argument("--prompt", default="billboard", 
                       help="Text prompt for detection")
    parser.add_argument("--detection-interval", type=int, default=30,
                       help="Interval for forced detection (frames)")
    parser.add_argument("--iou-threshold", type=float, default=0.6,
                       help="IoU threshold for drift detection")
    
    args = parser.parse_args()
    process_video_enhanced(args)

if __name__ == "__main__":
    main()