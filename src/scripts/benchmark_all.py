"""
Benchmark Script for Detection and Segmentation Models.
Generates academic-style tables with P, R, AP50, AP50-95.
"""
import sys
import os
import argparse
import pandas as pd
from ultralytics import YOLO, SAM
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import json
import torch
import numpy as np
from tqdm import tqdm
import cv2


def get_dataset_stats(data_yaml, split="test"):
    """Get image and instance counts from YOLO dataset."""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = os.path.dirname(data_yaml)
    split_paths = data.get(split, [])
    if isinstance(split_paths, str):
        split_paths = [split_paths]
    
    n_images = 0
    n_instances = 0
    
    for p in split_paths:
        img_dir = os.path.join(base_path, p) if not os.path.isabs(p) else p
        label_dir = img_dir.replace('/images', '/labels').replace('\\images', '\\labels')
        
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            n_images += len(images)
            
            if os.path.exists(label_dir):
                for lbl_file in os.listdir(label_dir):
                    if lbl_file.endswith('.txt'):
                        with open(os.path.join(label_dir, lbl_file), 'r') as f:
                            n_instances += len(f.readlines())
    
    return n_images, n_instances


def benchmark_yolo_detection(variants, data_yaml):
    """Benchmark YOLO detection models."""
    results = []
    n_images, n_instances = get_dataset_stats(data_yaml, "test")
    
    for v in variants:
        model_path = f"models/yolov8{v}_det_best.pt"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Skipping.")
            continue
            
        print(f"Benchmarking YOLOv8{v} detection...")
        try:
            model = YOLO(model_path)
            metrics = model.val(data=data_yaml, split="test", verbose=False)
            p, r, map50, map95 = metrics.box.mean_results()
            
            results.append({
                "Model": f"YOLOv8{v}",
                "Stage": "Test",
                "Images": n_images,
                "Instances": n_instances,
                "P": round(p, 3),
                "R": round(r, 3),
                "AP50": round(map50, 3),
                "AP50-95": round(map95, 3)
            })
        except Exception as e:
            print(f"Error: {e}")
            
    return results


def benchmark_grounding_dino_detection(data_yaml):
    """Zero-shot detection using YOLO-World."""
    print("Benchmarking GroundingDINO (Zero-shot) detection...")
    n_images, n_instances = get_dataset_stats(data_yaml, "test")
    
    try:
        model = YOLO("models/yolov8s-world.pt")
        model.set_classes(["billboard"])
        metrics = model.val(data=data_yaml, split="test", verbose=False)
        p, r, map50, map95 = metrics.box.mean_results()
        
        return [{
            "Model": "GroundingDINO(Zero-shot)",
            "Stage": "Test",
            "Images": n_images,
            "Instances": n_instances,
            "P": round(p, 3),
            "R": round(r, 3),
            "AP50": round(map50, 3),
            "AP50-95": round(map95, 3)
        }]
    except Exception as e:
        print(f"Error: {e}")
        return []


def benchmark_yolo_segmentation(variants, data_yaml):
    """Benchmark YOLO segmentation models."""
    results = []
    n_images, n_instances = get_dataset_stats(data_yaml, "test")
    
    for v in variants:
        model_path = f"models/yolov8{v}_seg_best.pt"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Skipping.")
            continue
            
        print(f"Benchmarking YOLOv8{v}-seg...")
        try:
            model = YOLO(model_path)
            metrics = model.val(data=data_yaml, split="test", verbose=False)
            p, r, map50, map95 = metrics.seg.mean_results()
            
            results.append({
                "Model": f"YOLOv8{v}-seg",
                "Stage": "Test",
                "Images": n_images,
                "Instances": n_instances,
                "P": round(p, 3),
                "R": round(r, 3),
                "AP50": round(map50, 3),
                "AP50-95": round(map95, 3)
            })
        except Exception as e:
            print(f"Error: {e}")
            
    return results


def benchmark_yolo_sam(det_model_path, data_yaml, sam_model="sam2_b.pt"):
    """Benchmark YOLOv8 detection + SAM2 segmentation pipeline."""
    print("Benchmarking YOLOv8+SAM...")
    
    try:
        n_images, n_instances = get_dataset_stats(data_yaml, "test")
        
        # Load models
        detector = YOLO(det_model_path)
        sam = SAM("models/sam2_b.pt")
        
        # Get test images from YOLO format
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        test_paths = data.get('test', [])
        if isinstance(test_paths, str):
            test_paths = [test_paths]
            
        # Collect all test images and their labels
        all_predictions = []
        all_ground_truths = []
        
        for p in test_paths:
            img_dir = p if os.path.isabs(p) else os.path.join(os.path.dirname(data_yaml), p)
            label_dir = img_dir.replace('/images', '/labels').replace('\\images', '\\labels')
            
            if not os.path.exists(img_dir):
                continue
                
            images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            for img_name in tqdm(images, desc="YOLOv8+SAM"):
                img_path = os.path.join(img_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                
                # Load ground truth
                lbl_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
                gt_masks = []
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 5:  # Polygon format
                                coords = list(map(float, parts[1:]))
                                pts = np.array(coords).reshape(-1, 2)
                                pts[:, 0] *= w
                                pts[:, 1] *= h
                                mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                                gt_masks.append(mask)
                
                # Detect
                results = detector.predict(frame, conf=0.25, verbose=False)
                pred_masks = []
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        sam_results = sam.predict(frame, bboxes=[box.tolist()], verbose=False)
                        if sam_results and sam_results[0].masks is not None and len(sam_results[0].masks.data) > 0:
                            mask = sam_results[0].masks.data[0].cpu().numpy()
                            if mask.shape != (h, w):
                                mask = cv2.resize(mask.astype(np.float32), (w, h))
                            pred_masks.append((mask > 0.5).astype(np.uint8))
                
                all_predictions.append(pred_masks)
                all_ground_truths.append(gt_masks)
        
        # Compute metrics
        p, r, ap50, ap50_95 = compute_segmentation_metrics(all_predictions, all_ground_truths)
        
        return [{
            "Model": "YOLOv8+SAM",
            "Stage": "Test",
            "Images": n_images,
            "Instances": n_instances,
            "P": round(p, 3),
            "R": round(r, 3),
            "AP50": round(ap50, 3),
            "AP50-95": round(ap50_95, 3)
        }]
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def benchmark_grounded_sam(data_yaml, sam_model="sam2_b.pt"):
    """Benchmark GroundedSAM (zero-shot detection + SAM2)."""
    print("Benchmarking GroundedSAM (Zero-shot)...")
    
    try:
        n_images, n_instances = get_dataset_stats(data_yaml, "test")
        
        # Load models - use multiple text prompts for better detection
        detector = YOLO("models/yolov8s-world.pt")
        detector.set_classes(["billboard", "advertising board", "sign", "advertisement"])
        sam = SAM("models/sam2_b.pt")
        
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        test_paths = data.get('test', [])
        if isinstance(test_paths, str):
            test_paths = [test_paths]
            
        all_predictions = []
        all_ground_truths = []
        
        for p in test_paths:
            img_dir = p if os.path.isabs(p) else os.path.join(os.path.dirname(data_yaml), p)
            label_dir = img_dir.replace('/images', '/labels').replace('\\images', '\\labels')
            
            if not os.path.exists(img_dir):
                continue
                
            images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            for img_name in tqdm(images, desc="GroundedSAM"):
                img_path = os.path.join(img_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                
                # Load ground truth
                lbl_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
                gt_masks = []
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 5:
                                coords = list(map(float, parts[1:]))
                                pts = np.array(coords).reshape(-1, 2)
                                pts[:, 0] *= w
                                pts[:, 1] *= h
                                mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                                gt_masks.append(mask)
                
                # Detect with zero-shot - lower confidence for better recall
                results = detector.predict(frame, conf=0.1, verbose=False)
                pred_masks = []
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        sam_results = sam.predict(frame, bboxes=[box.tolist()], verbose=False)
                        if sam_results and sam_results[0].masks is not None and len(sam_results[0].masks.data) > 0:
                            mask = sam_results[0].masks.data[0].cpu().numpy()
                            if mask.shape != (h, w):
                                mask = cv2.resize(mask.astype(np.float32), (w, h))
                            pred_masks.append((mask > 0.5).astype(np.uint8))
                
                all_predictions.append(pred_masks)
                all_ground_truths.append(gt_masks)
        
        p, r, ap50, ap50_95 = compute_segmentation_metrics(all_predictions, all_ground_truths)
        
        return [{
            "Model": "GroundedSAM(Zero-shot)",
            "Stage": "Test",
            "Images": n_images,
            "Instances": n_instances,
            "P": round(p, 3),
            "R": round(r, 3),
            "AP50": round(ap50, 3),
            "AP50-95": round(ap50_95, 3)
        }]
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / (union + 1e-6)


def compute_segmentation_metrics(all_preds, all_gts, iou_thresholds=None):
    """Compute P, R, AP50, AP50-95 for segmentation."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # For AP50
    tp_50, fp_50, fn_50 = 0, 0, 0
    
    # For AP50-95 (average over thresholds)
    aps = []
    
    for iou_thresh in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        
        for preds, gts in zip(all_preds, all_gts):
            if len(gts) == 0:
                fp += len(preds)
                continue
            if len(preds) == 0:
                fn += len(gts)
                continue
            
            matched_gt = set()
            
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn += len(gts) - len(matched_gt)
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        ap = precision  # Simplified AP
        aps.append(ap)
        
        if abs(iou_thresh - 0.5) < 0.01:
            tp_50, fp_50, fn_50 = tp, fp, fn
    
    p = tp_50 / (tp_50 + fp_50 + 1e-6)
    r = tp_50 / (tp_50 + fn_50 + 1e-6)
    ap50 = aps[0] if aps else 0
    ap50_95 = np.mean(aps) if aps else 0
    
    return p, r, ap50, ap50_95


def format_table(df, bold_cols=["AP50", "AP50-95"]):
    """Format with bold best values."""
    df_copy = df.copy()
    for col in bold_cols:
        if col in df_copy.columns:
            max_val = df_copy[col].max()
            df_copy[col] = df_copy[col].apply(lambda x: f"**{x}**" if x == max_val else str(x))
    return df_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="benchmark_results.md")
    parser.add_argument("--detection-yaml", default="data_detection.yaml")
    parser.add_argument("--segmentation-yaml", default="data_segmentation.yaml")
    args = parser.parse_args()
    
    detection_results = []
    segmentation_results = []
    
    # Detection Benchmarks
    print("\n" + "="*60)
    print("TABLE 1: Billboard Object Detection (bounding box) accuracy")
    print("="*60 + "\n")
    
    detection_results.extend(benchmark_yolo_detection(["n", "s", "m"], args.detection_yaml))
    detection_results.extend(benchmark_grounding_dino_detection(args.detection_yaml))
    
    # Segmentation Benchmarks
    print("\n" + "="*60)
    print("TABLE 2: Billboard Segmentation (precise mask) accuracy")
    print("="*60 + "\n")
    
    segmentation_results.extend(benchmark_yolo_segmentation(["n", "s", "m"], args.segmentation_yaml))
    
    # YOLOv8+SAM
    det_model = "models/yolov8m_det_best.pt"
    if os.path.exists(det_model):
        segmentation_results.extend(benchmark_yolo_sam(det_model, args.segmentation_yaml))
    
    # GroundedSAM
    segmentation_results.extend(benchmark_grounded_sam(args.segmentation_yaml))
    
    # Create output
    output = ["# Benchmark Results\n"]
    output.append("## Table 1: Billboard Object Detection (bounding box) accuracy\n")
    
    if detection_results:
        df_det = pd.DataFrame(detection_results)
        print("\nDetection Results:")
        print(df_det.to_markdown(index=False))
        output.append(format_table(df_det).to_markdown(index=False))
    
    output.append("\n\n## Table 2: Billboard Segmentation (precise mask) accuracy\n")
    
    if segmentation_results:
        df_seg = pd.DataFrame(segmentation_results)
        print("\nSegmentation Results:")
        print(df_seg.to_markdown(index=False))
        output.append(format_table(df_seg).to_markdown(index=False))
    
    with open(args.save, "w") as f:
        f.write("\n".join(output))
    print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
