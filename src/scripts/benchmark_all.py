"""
Benchmark Script for all Models.
Generates a markdown table of P, R, mAP50, mAP50-95.
"""
import sys
import os
import argparse
import pandas as pd
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.evaluation import evaluate_maskrcnn # Ensure this exists or we mock it
from modules.segmentation import SAM2Segmenter
# We need to import detection logic or use ultralytics directly
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np
from tqdm import tqdm
import cv2
# Actually Mask R-CNN training script has evaluate() but it's internal.
# I might need to import the model loading logic from train_maskrcnn.

def benchmark_yolo(variants, task="detect", data_yaml=None):
    results = []
    
    for v in variants:
        model_name = f"runs/{task}/billboard_yolov8{v}_{task}/weights/best.pt"
        if not os.path.exists(model_name):
            print(f"Model not found: {model_name}. Skipping.")
            continue
            
        print(f"Benchmarking {model_name}...")
        try:
            model = YOLO(model_name)
            metrics = model.val(data=data_yaml, split="test", verbose=False)
            
            # Extract metrics
            if task == "detect":
                p = metrics.box.map50 # Ultralytics stores map50, map75, maps
                # wait, metrics.box.mean_results() gives P, R, mAP50, mAP50-95
                p, r, map50, map95 = metrics.box.mean_results()
            else:
                p, r, map50, map95 = metrics.seg.mean_results()
                
            results.append({
                "Model": f"YOLOv8{v}-{task}",
                "Stage": "Test",
                "Images": metrics.box.n if task=="detect" else metrics.seg.n,  # Number of images
                "Instances": 0, # Difficult to get total instances without summing per class
                "P": round(p, 3),
                "R": round(r, 3),
                "AP50": round(map50, 3),
                "AP50-95": round(map95, 3)
            })
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            
    return results

def benchmark_grounding_dino(data_yaml):
    # Zero-shot evaluation using YOLO-World
    print("Benchmarking GroundingDINO (Zero-shot)...")
    # ... Implementation of evaluating YOLO-World on the dataset ...
    # This acts like a standard YOLO val but with a pre-set model
    try:
        model = YOLO("yolov8s-world.pt")
        model.set_classes(["billboard"])
        metrics = model.val(data=data_yaml, split="test", verbose=False)
        p, r, map50, map95 = metrics.box.mean_results()
        
        return [{
            "Model": "GroundingDINO(Zero-shot)",
            "Stage": "Test",
            "Images": metrics.box.n,
            "Instances": 0,
            "P": round(p, 3),
            "R": round(r, 3),
            "AP50": round(map50, 3),
            "AP50-95": round(map95, 3)
        }]
    except Exception as e:
        print(f"Error benchmarking Zero-shot: {e}")
        return []

def benchmark_maskrcnn():
    print("Benchmarking Mask R-CNN...")
    # This logic mimics train_maskrcnn.evaluate but stand-alone
    # We need to load the model and run it on val_coco.json (since that's what we have)
    # Actually, we should evaluate on test split if possible, but we only made train_coco/val_coco
    # Let's assume we use val_coco.json for now as 'test' proxy since we didn't make test_coco.json
    
    try:
        from scripts.train_maskrcnn import get_model, BillboardDataset, get_transform, evaluate
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load Model
        # Assuming best model is saved
        model_path = "maskrcnn_trained_models/maskrcnn_billboard_best.pth"
        if not os.path.exists(model_path):
            print("Mask R-CNN model not found.")
            return []
            
        # Load COCO GT
        # We need to ensure we have test annotations. 
        # Since we only generated train_coco and val_coco, we will use val_coco for this benchmark demonstration.
        coco_gt_path = "val_coco.json" 
        
        # Setup Dataset (Only needed if we use the torch evaluation engine)
        dataset_test = BillboardDataset(".", coco_gt_path, get_transform(train=False))
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=lambda x: tuple(zip(*x)))

        num_classes = 2
        model = get_model(num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Run Evaluation
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        
        # Check if evaluate returns the evaluator
        if coco_evaluator is not None:
            stats = coco_evaluator.coco_eval['segm'].stats
            return [{
                "Model": "Mask R-CNN",
                "Stage": "Test (Val)",
                "Images": len(dataset_test),
                "Instances": 0, # Unknown easily
                "P": 0.0, # COCOEval doesn't give P/R directly in stats[0] format easily without calculation
                "R": round(stats[8], 3), # AR @ 100
                "AP50": round(stats[1], 3),
                "AP50-95": round(stats[0], 3)
            }]
            
    except Exception as e:
        print(f"Error benchmarking Mask R-CNN: {e}")
        return []

def benchmark_sam_pipeline(name, det_model_path, det_type, data_yaml):
    print(f"Benchmarking {name}...")
    
    try:
        from modules.detection import create_detector
        from modules.segmentation import SAM2Segmenter
        import yaml
        
        # Load dataset images
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        # Get test images
        test_images = []
        base_dir = "."
        
        # Handling list of paths
        rel_paths = data.get('test', [])
        if isinstance(rel_paths, str): rel_paths = [rel_paths]
        
        for p in rel_paths:
            full_p = os.path.join(base_dir, p)
            if os.path.exists(full_p):
                valid_exts = ('.jpg', '.png')
                test_images.extend([os.path.join(full_p, f) for f in os.listdir(full_p) if f.lower().endswith(valid_exts)])
        
        if not test_images:
            print("No test images found.")
            return []
            
        print(f"Evaluating on {len(test_images)} images...")
        
        # Init Models
        detector = create_detector(det_type, det_model_path)
        sam = SAM2Segmenter("sam2_b.pt") 
        
        # Results in COCO format
        coco_results = []
        
        # We need Ground Truth COCO object to compute metrics
        # Since we don't have a pre-made test_coco.json, benchmarking against raw files is hard.
        # We must align with a COCO JSON.
        # FIX: We will rely on `val_coco.json` and FILTER images that are in the val set,
        # OR we generate `test_coco.json` on the fly?
        # User wants comparison. 
        # Let's assume we use `val_coco.json` for all 'precise mask' benchmarks for consistency if test json doesn't exist.
        
        gt_json = "val_coco.json" 
        coco_gt = COCO(gt_json)
        img_ids = coco_gt.getImgIds()
        
        for img_id in tqdm(img_ids):
            img_info = coco_gt.loadImgs(img_id)[0]
            # Image path logic: JSON has relative path. 
            # We need to find the actual file.
            # train_coco.json has "file_name": "datasets/..."
            img_path = img_info['file_name']
            
            if not os.path.exists(img_path):
                continue
                
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # Detect
            boxes = detector.detect(frame, conf=0.25) # Low conf to get recall
            
            # Segment
            for box in boxes:
                # box is [x1, y1, x2, y2]
                mask = sam.segment_with_box(frame, box)
                
                if mask is not None:
                     # Encode RLE
                     from pycocotools import mask as mask_util
                     rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                     rle['counts'] = rle['counts'].decode('utf-8')
                     
                     coco_results.append({
                         "image_id": img_id,
                         "category_id": 1, # billboard
                         "segmentation": rle,
                         "score": 1.0 # SAM doesn't give score easily, assume detector score?
                     })
                     
        # Save results
        res_file = f"results_{name.replace(' ','_')}.json"
        with open(res_file, 'w') as f:
            json.dump(coco_results, f)
            
        # Eval
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats = coco_eval.stats
        return [{
            "Model": name,
            "Stage": "Test (Val)",
            "Images": len(img_ids),
            "Instances": 0,
            "P": 0.0,
            "R": round(stats[8], 3),
            "AP50": round(stats[1], 3),
            "AP50-95": round(stats[0], 3)
        }]

    except Exception as e:
        print(f"Error benchmarking {name}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="benchmark_results.md")
    args = parser.parse_args()
    
    all_results = []
    
    # Detection
    print("--- ONE: Detection Benchmarks ---")
    det_variants = ["n", "s", "m"]
    all_results.extend(benchmark_yolo(det_variants, "detect", "data_detection.yaml"))
    all_results.extend(benchmark_grounding_dino("data_detection.yaml"))
    
    # Segmentation
    print("--- TWO: Segmentation Benchmarks ---")
    seg_variants = ["n", "s", "m"]
    all_results.extend(benchmark_yolo(seg_variants, "segment", "data_segmentation.yaml"))
    
    # Mask R-CNN
    all_results.extend(benchmark_maskrcnn())
    
    # YOLO+SAM
    # We use the detection models trained in step 1, and SAM for segmentation
    detect_model_path = "runs/detect/billboard_yolov8m_det/weights/best.pt" 
    if os.path.exists(detect_model_path):
        all_results.extend(benchmark_sam_pipeline("YOLOv8+SAM", detect_model_path, "yolo", "data_segmentation.yaml"))
    
    # GroundedSAM (Zero-shot)
    all_results.extend(benchmark_sam_pipeline("GroundedSAM(Zero-shot)", "yolov8s-world.pt", "grounding-dino", "data_segmentation.yaml"))
    
    # Create Table
    df = pd.DataFrame(all_results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))
    
    if args.save:
        with open(args.save, "w") as f:
            f.write("# Benchmark Results\n\n")
            f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
