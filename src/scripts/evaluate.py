import sys
import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.detection import BillboardDetector
from modules.segmentation import GroundedSAM

def compute_iou(mask1, mask2):
    """Computes IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score if np.sum(union) > 0 else 0.0

def evaluate_fine_tuned(model_path="runs/segment/billboard_segmentation5/weights/best.pt", data_yaml="data.yaml"):
    """
    Evaluates fine-tuned YOLOv8-seg using built-in validator.
    """
    print(f"Evaluating Fine-Tuned Model: {model_path}...")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using 'yolov8n-seg.pt' for demo.")
        model_path = "yolov8n-seg.pt"
        
    model = YOLO(model_path)
    # Using 'valid' split for now as test might not be labeled or pointing to same
    metrics = model.val(data=data_yaml, split='val', verbose=False) # 'test' if available
    
    print("Fine-Tuned Metrics:")
    print(f"Box mAP50-95: {metrics.box.map}")
    print(f"Seg mAP50-95: {metrics.seg.map}")
    return metrics

def evaluate_zero_shot(grounded_sam, image_dir="valid/images", label_dir="valid/labels"):
    """
    Evaluates GroundedSAM by computing mean IoU on the validation set.
    """
    print("\nEvaluating Zero-Shot Model (GroundedSAM)...")
    
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    ious = []
    
    print(f"Found {len(image_paths)} images for Zero-Shot evaluation.")
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # Ground Truth Loading (Simplification: assuming 1 object or using first)
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(label_path):
            continue
            
        # Parse YOLO label to mask
        h, w = frame.shape[:2]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                
                if len(parts) == 5:
                    # Bounding Box: class x_center y_center w h
                    xc, yc, bw, bh = parts[1:]
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    cv2.rectangle(gt_mask, (x1, y1), (x2, y2), 1, -1)
                else:
                    # Polygon
                    points = np.array(parts[1:]).reshape(-1, 2)
                    points[:, 0] *= w
                    points[:, 1] *= h
                    cv2.fillPoly(gt_mask, [points.astype(np.int32)], 1)
        
        # Prediction
        masks, bboxes = grounded_sam.detect_and_segment(frame, text_prompt="billboard")
        
        if len(masks) == 0:
            # Try specific Turkish prompt or different synonym?
            # masks, bboxes = grounded_sam.detect_and_segment(frame, text_prompt="screen")
            ious.append(0.0)
        else:
            # Taking max IoU among predictions
            frame_ious = []
            for pred_mask in masks:
                # Resize if needed (SAM usually returns native resolution but check)
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                score = compute_iou(gt_mask, pred_mask)
                frame_ious.append(score)
            max_iou = max(frame_ious)
            ious.append(max_iou)
            # print(f"Image {os.path.basename(img_path)}: IoU {max_iou:.4f}")

    mean_iou = np.mean(ious) if ious else 0.0
    print(f"Zero-Shot Mean IoU: {mean_iou:.4f}")
    return mean_iou

if __name__ == "__main__":
    # Check if models exist
    # Evaluate Fine-Tuned
    try:
        evaluate_fine_tuned()
    except Exception as e:
        print(f"Fine-tuned evaluation failed: {e}")

    # Evaluate Zero-Shot
    try:
        gs = GroundedSAM(sam_model_path="sam2_b.pt", grounding_model_path="yolov8s-world.pt")
        evaluate_zero_shot(gs)
    except Exception as e:
        print(f"Zero-shot evaluation failed: {e}")
