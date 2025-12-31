"""
Evaluation Module for Billboard Segmentation.

Metrics:
- IoU (Intersection over Union)
"""
import numpy as np

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate IoU between predicted and ground truth masks.
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
        
    Returns:
        IoU score (0.0 - 1.0)
    """
    # Ensure binary masks
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    
    if union == 0:
        return 0.0
    
    return intersection / union
