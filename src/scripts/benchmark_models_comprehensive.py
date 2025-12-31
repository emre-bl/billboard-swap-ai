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
from PIL import Image
import json
import torch
import numpy as np
from tqdm import tqdm
import cv2
import tempfile
import shutil
import torchvision.transforms as T


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


def benchmark_yolo_world_detection(data_yaml):
    """Zero-shot detection using YOLO-Worldv2."""
    print("Benchmarking YOLO-Worldv2 (Zero-shot) detection...")
    n_images, n_instances = get_dataset_stats(data_yaml, "test")
    
    try:
        # Use YOLOv8m-worldv2 for zero-shot detection
        model = YOLO("yolov8m-worldv2.pt")
        model.set_classes(["billboard"])
        metrics = model.val(data=data_yaml, split="test", verbose=False)
        p, r, map50, map95 = metrics.box.mean_results()
        
        return [{
            "Model": "YOLO-Worldv2(Zero-shot)",
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
        import traceback
        traceback.print_exc()
        return []


def compute_detection_metrics(all_preds, all_gts, iou_thresholds=None):
    """Compute P, R, AP50, AP50-95 for detection using box IoU."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    def box_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    aps = []
    tp_50, fp_50, fn_50 = 0, 0, 0
    
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
                    iou = box_iou(pred, gt)
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
        aps.append(precision)
        
        if abs(iou_thresh - 0.5) < 0.01:
            tp_50, fp_50, fn_50 = tp, fp, fn
    
    p = tp_50 / (tp_50 + fp_50 + 1e-6)
    r = tp_50 / (tp_50 + fn_50 + 1e-6)
    ap50 = aps[0] if aps else 0
    ap50_95 = np.mean(aps) if aps else 0
    
    return p, r, ap50, ap50_95




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


def compute_coco_metrics_from_predictions(gt_data_yaml, predictions_json):
    """
    Compute official COCO metrics from predictions JSON file using gt from YAML.
    
    Args:
        gt_data_yaml: Path to data.yaml
        predictions_json: Path to COCO-format predictions JSON
    
    Returns:
        tuple: (p, r, ap50, ap95)
    """
    with open(gt_data_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    test_paths = data.get('test', [])
    if isinstance(test_paths, str):
        test_paths = [test_paths]
        
    # 1. Create Ground Truth COCO JSON
    gt_coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "billboard"}]
    }
    
    ann_id = 0
    img_id_map = {} # path -> id
    
    # Collect all images and annotations first
    img_files = []
    
    for p in test_paths:
        img_dir = p if os.path.isabs(p) else os.path.join(os.path.dirname(gt_data_yaml), p)
        if not os.path.exists(img_dir): continue
        files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        img_files.extend(files)

    for i, img_path in enumerate(img_files):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        w, h = img.size
        
        img_id = i + 1
        img_id_map[img_path] = img_id
        
        gt_coco_dict["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        # Load labels
        label_dir = os.path.dirname(img_path).replace('images', 'labels')
        lbl_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 5:  # Polygon
                        coords = list(map(float, parts[1:]))
                        # Convert normalized to absolute
                        poly = []
                        for j in range(0, len(coords), 2):
                            poly.append(coords[j] * w)
                            poly.append(coords[j+1] * h)
                        
                        # RLE encoding for area/bbox
                        rle = mask_util.frPyObjects([poly], h, w)[0]
                        bbox = mask_util.toBbox(rle).tolist()
                        area = mask_util.area(rle)
                        
                        gt_coco_dict["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "segmentation": [poly],
                            "area": float(area),
                            "bbox": bbox,
                            "iscrowd": 0
                        })
                        ann_id += 1
    
    # Save temp GT JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_coco_dict, f)
        gt_json_path = f.name
        
    try:
        # 2. Run COCO Evaluation
        cocoGt = COCO(gt_json_path)
        
        # Load predictions and map file_names back to IDs if needed (or assume ID usage)
        # The benchmark functions below should produce predictions with correct image_id
        cocoDt = cocoGt.loadRes(predictions_json)
        
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        stats = cocoEval.stats
        ap50_95 = stats[0]
        ap50 = stats[1]
        
        # Note: COCOEval does not output P/R directly in standard stats array in the same way YOLO does
        # But we can approximate or just use mAP as the gold standard.
        # However, for consistency with table, we can extract P/R from precision array if desired,
        # or stick to AP.
        # YOLO typically reports P, R at best confidence. COCO reports mAP.
        
        # We will return AP50 and AP50-95. P/R might be less compatible directly from COCO API without curve analysis.
        # But we can try to get them from accumulated stats if needed, or just report 0 for now.
        return 0, 0, ap50, ap50_95
        
    except Exception as e:
        print(f"COCO Eval Error: {e}")
        return 0, 0, 0, 0
    finally:
        os.remove(gt_json_path)


def benchmark_yolo_sam_coco(det_model_path, data_yaml, sam_model="sam2_b.pt"):
    """Benchmark YOLOv8+SAM2 using official COCO evaluation."""
    print("Benchmarking YOLOv8+SAM (COCO Eval)...")
    
    try:
        n_images, n_instances = get_dataset_stats(data_yaml, "test")
        
        detector = YOLO(det_model_path)
        sam = SAM("models/sam2_b.pt")
        
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        test_paths = data.get('test', [])
        if isinstance(test_paths, str): test_paths = [test_paths]
        
        img_files = []
        for p in test_paths:
            img_dir = p if os.path.isabs(p) else os.path.join(os.path.dirname(data_yaml), p)
            if not os.path.exists(img_dir): continue
            img_files.extend(sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]))

        predictions = []
        
        for i, img_path in tqdm(enumerate(img_files), desc="YOLOv8+SAM Inference"):
            frame = cv2.imread(img_path)
            if frame is None: continue
            h, w = frame.shape[:2]
            img_id = i + 1  # 1-based index to match GT generation above
            
            # Detect
            results = detector.predict(frame, conf=0.25, verbose=False)
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    # SAM Inference
                    sam_results = sam.predict(frame, bboxes=[box.tolist()], verbose=False)
                    if sam_results and sam_results[0].masks is not None:
                        # Select Best Mask
                        masks_data = sam_results[0].masks.data.cpu().numpy()
                        best_idx = 0
                        
                        # Use SAM scores if available, else area
                        if hasattr(sam_results[0], 'scores') and sam_results[0].scores is not None:
                             mask_scores = sam_results[0].scores.cpu().numpy()
                             best_idx = np.argmax(mask_scores)
                        
                        mask = masks_data[best_idx]
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        
                        # To RLE for COCO JSON
                        rle = mask_util.encode(np.asfortranarray(mask_binary))
                        rle['counts'] = rle['counts'].decode('utf-8')
                        
                        predictions.append({
                            "image_id": img_id,
                            "category_id": 0,
                            "segmentation": rle,
                            "score": float(score) # Use detection score as proxy
                        })

        # Save predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_json_path = f.name
            
        # Run Evaluation
        _, _, ap50, ap50_95 = compute_coco_metrics_from_predictions(data_yaml, pred_json_path)
        os.remove(pred_json_path)

        return [{
            "Model": "YOLOv8+SAM2",
            "Stage": "Test",
            "Images": n_images,
            "Instances": n_instances,
            "P": "-", # COCOEval makes P/R hard to extract simply
            "R": "-",
            "AP50": round(ap50, 3),
            "AP50-95": round(ap50_95, 3)
        }]
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- MASK R-CNN UTILS ---
def get_maskrcnn_model(num_classes: int = 2):
    """Create Mask R-CNN model with custom head."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

def benchmark_maskrcnn(data_yaml):
    """Benchmark Mask R-CNN using official COCO evaluation."""
    print("Benchmarking Mask R-CNN (COCO Eval)...")
    model_path = "maskrcnn_trained_models/best.pth"
    if not os.path.exists(model_path):
        print(f"Mask R-CNN model not found at {model_path}")
        return []
        
    try:
        n_images, n_instances = get_dataset_stats(data_yaml, "test")
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_maskrcnn_model(2)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Get test images
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        test_paths = data.get('test', [])
        if isinstance(test_paths, str): test_paths = [test_paths]
            
        img_files = []
        for p in test_paths:
            img_dir = p if os.path.isabs(p) else os.path.join(os.path.dirname(data_yaml), p)
            if not os.path.exists(img_dir): continue
            img_files.extend(sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]))

        predictions = []
        
        for i, img_path in tqdm(enumerate(img_files), desc="Mask R-CNN Inference"):
            # Load and Transform
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = T.functional.to_tensor(img_pil).to(device)
            img_id = i + 1
            
            # Predict
            with torch.no_grad():
                prediction = model([img_tensor])[0]
            
            masks = prediction['masks'].squeeze(1).cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for j, score in enumerate(scores):
                if score > 0.05: # Slight threshold to reduce JSON size
                    mask = (masks[j] > 0.5).astype(np.uint8)
                    
                    # To RLE
                    rle = mask_util.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    
                    predictions.append({
                        "image_id": img_id,
                        "category_id": 0,
                        "segmentation": rle,
                        "score": float(score)
                    })

        # Save predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_json_path = f.name
            
        # Run Evaluation
        _, _, ap50, ap50_95 = compute_coco_metrics_from_predictions(data_yaml, pred_json_path)
        os.remove(pred_json_path)
        
        return [{
            "Model": "Mask R-CNN",
            "Stage": "Test",
            "Images": n_images,
            "Instances": n_instances,
            "P": "-",
            "R": "-",
            "AP50": round(ap50, 3),
            "AP50-95": round(ap50_95, 3)
        }]
        
    except Exception as e:
        print(f"Error in Mask R-CNN benchmark: {e}")
        import traceback
        traceback.print_exc()
        return []


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
    detection_results.extend(benchmark_yolo_world_detection(args.detection_yaml))
    
    # Segmentation Benchmarks
    print("\n" + "="*60)
    print("TABLE 2: Billboard Segmentation (precise mask) accuracy")
    print("="*60 + "\n")
    
    segmentation_results.extend(benchmark_yolo_segmentation(["n", "s", "m"], args.segmentation_yaml))
    
    # YOLOv8+SAM
    det_model = "models/yolov8m_det_best.pt"
    if os.path.exists(det_model):
        segmentation_results.extend(benchmark_yolo_sam_coco(det_model, args.segmentation_yaml))
    
    # Mask R-CNN
    segmentation_results.extend(benchmark_maskrcnn(args.segmentation_yaml))
    
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
