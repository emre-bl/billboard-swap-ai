"""
Evaluation Module for Billboard Segmentation.

Metrics:
- IoU (Intersection over Union)
- mAP (mean Average Precision) at IoU thresholds 0.5 and 0.5-0.95
- Precision, Recall, F1-Score
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Single detection result."""
    mask: np.ndarray
    confidence: float
    class_id: int = 0


@dataclass 
class EvaluationMetrics:
    """Evaluation metrics container."""
    precision: float
    recall: float
    f1_score: float
    iou: float
    ap50: float
    ap50_95: float
    

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


def calculate_precision_recall(
    predictions: List[DetectionResult],
    ground_truths: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate precision and recall at given IoU threshold.
    
    Args:
        predictions: List of detection results
        ground_truths: List of ground truth masks
        iou_threshold: IoU threshold for true positive
        
    Returns:
        Tuple of (precision, recall)
    """
    if not predictions:
        return 0.0, 0.0
    
    if not ground_truths:
        return 0.0, 0.0
    
    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
    
    # Track which GTs have been matched
    gt_matched = [False] * len(ground_truths)
    
    true_positives = 0
    false_positives = 0
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_mask in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou(pred.mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            gt_matched[best_gt_idx] = True
        else:
            false_positives += 1
    
    false_negatives = sum(1 for matched in gt_matched if not matched)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


def calculate_ap(
    predictions: List[DetectionResult],
    ground_truths: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision at given IoU threshold.
    
    Args:
        predictions: List of detection results
        ground_truths: List of ground truth masks
        iou_threshold: IoU threshold
        
    Returns:
        AP score
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # Sort by confidence
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
    
    gt_matched = [False] * len(ground_truths)
    
    precisions = []
    recalls = []
    
    tp = 0
    fp = 0
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_mask in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou(pred.mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
        
        precision = tp / (tp + fp)
        recall = tp / len(ground_truths)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_at_recall:
            ap += max(precisions_at_recall) / 11
    
    return ap


def calculate_map(
    predictions: List[DetectionResult],
    ground_truths: List[np.ndarray],
    iou_thresholds: Tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
) -> Dict[str, float]:
    """
    Calculate mAP at various IoU thresholds.
    
    Args:
        predictions: List of detection results
        ground_truths: List of ground truth masks
        iou_thresholds: Tuple of IoU thresholds
        
    Returns:
        Dict with mAP@50 and mAP@50-95
    """
    # AP@50
    ap50 = calculate_ap(predictions, ground_truths, 0.5)
    
    # AP@50-95 (average over thresholds)
    aps = [calculate_ap(predictions, ground_truths, t) for t in iou_thresholds]
    ap50_95 = np.mean(aps)
    
    return {
        "mAP@50": ap50,
        "mAP@50-95": ap50_95,
    }


def evaluate_segmenter(
    segmenter,
    images: List[np.ndarray],
    ground_truths: List[List[np.ndarray]],
    confidence_threshold: float = 0.5
) -> EvaluationMetrics:
    """
    Evaluate a segmenter on a set of images.
    
    Args:
        segmenter: BaseSegmenter instance
        images: List of BGR images
        ground_truths: List of lists of GT masks per image
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        EvaluationMetrics
    """
    all_predictions = []
    all_ground_truths = []
    
    total_iou = 0
    num_matches = 0
    
    for img, gts in zip(images, ground_truths):
        # Get predictions
        masks = segmenter.segment(img, conf=confidence_threshold)
        
        # Convert to DetectionResult
        for mask in masks:
            all_predictions.append(DetectionResult(
                mask=mask,
                confidence=1.0,  # Assume max confidence if not available
                class_id=0
            ))
        
        all_ground_truths.extend(gts)
        
        # Calculate IoU for best matches
        for gt in gts:
            if masks:
                ious = [calculate_iou(m, gt) for m in masks]
                total_iou += max(ious)
                num_matches += 1
    
    # Calculate metrics
    precision, recall = calculate_precision_recall(
        all_predictions, all_ground_truths, iou_threshold=0.5
    )
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_iou = total_iou / num_matches if num_matches > 0 else 0
    
    map_metrics = calculate_map(all_predictions, all_ground_truths)
    
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        iou=avg_iou,
        ap50=map_metrics["mAP@50"],
        ap50_95=map_metrics["mAP@50-95"]
    )


def print_metrics(metrics: EvaluationMetrics, model_name: str = "Model"):
    """Print evaluation metrics in formatted table."""
    print(f"\n{'='*50}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Precision:  {metrics.precision:.4f}")
    print(f"  Recall:     {metrics.recall:.4f}")
    print(f"  F1-Score:   {metrics.f1_score:.4f}")
    print(f"  IoU:        {metrics.iou:.4f}")
    print(f"  mAP@50:     {metrics.ap50:.4f}")
    print(f"  mAP@50-95:  {metrics.ap50_95:.4f}")
    print(f"{'='*50}")
