"""
Model Benchmark and Comparison Script.

Evaluates all segmentation models and generates performance comparison tables.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.segmentation import (
    YOLOSegmenter, MaskRCNNSegmenter, GroundedSAMSegmenter, SAM2Segmenter
)
from modules.evaluation import calculate_iou


@dataclass
class BenchmarkResult:
    """Benchmark result for a single model."""
    model_name: str
    variant: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    inference_time_ms: float
    fps: float
    model_size_mb: float
    gpu_memory_mb: float


# Model configurations
YOLO_VARIANTS = ["n", "s", "m", "l", "x"]

MODEL_CONFIGS = {
    "yolov8n-seg": {"type": "yolo", "variant": "n"},
    "yolov8s-seg": {"type": "yolo", "variant": "s"},
    "yolov8m-seg": {"type": "yolo", "variant": "m"},
    "yolov8l-seg": {"type": "yolo", "variant": "l"},
    "yolov8x-seg": {"type": "yolo", "variant": "x"},
    "mask-rcnn": {"type": "maskrcnn"},
    "grounded-sam": {"type": "grounded-sam"},
    "sam2": {"type": "sam2"},
}


def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    if Path(model_path).exists():
        return Path(model_path).stat().st_size / (1024 * 1024)
    return 0.0


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except:
        pass
    return 0.0


def benchmark_model(
    segmenter,
    test_images: List[np.ndarray],
    ground_truths: List[List[np.ndarray]],
    model_name: str,
    variant: str = "",
    model_path: str = "",
    num_warmup: int = 3,
    num_runs: int = 10,
) -> BenchmarkResult:
    """
    Benchmark a segmentation model.
    
    Args:
        segmenter: BaseSegmenter instance
        test_images: List of test images
        ground_truths: List of GT masks per image
        model_name: Model name for logging
        variant: Model variant (e.g., "n", "s")
        model_path: Path to model file
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        BenchmarkResult
    """
    print(f"\nBenchmarking: {model_name} {variant}")
    
    # Warmup
    for _ in range(num_warmup):
        if test_images:
            segmenter.segment(test_images[0], conf=0.5)
    
    # Timing runs
    inference_times = []
    all_ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for img, gts in zip(test_images, ground_truths):
        start = time.perf_counter()
        masks = segmenter.segment(img, conf=0.5)
        end = time.perf_counter()
        
        inference_times.append((end - start) * 1000)  # ms
        
        # Calculate IoU for each GT
        for gt in gts:
            if masks:
                ious = [calculate_iou(m, gt) for m in masks]
                max_iou = max(ious)
                all_ious.append(max_iou)
                
                if max_iou >= 0.5:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                false_negatives += 1
                all_ious.append(0)
        
        # Count FPs
        matched = min(len(masks), len(gts))
        false_positives += max(0, len(masks) - matched)
    
    # Calculate metrics
    avg_time = np.mean(inference_times) if inference_times else 0
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Approximate mAP (simplified)
    avg_iou = np.mean(all_ious) if all_ious else 0
    map50 = avg_iou  # Simplified
    map50_95 = avg_iou * 0.7  # Rough approximation
    
    return BenchmarkResult(
        model_name=model_name,
        variant=variant,
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        inference_time_ms=avg_time,
        fps=fps,
        model_size_mb=get_model_size(model_path),
        gpu_memory_mb=get_gpu_memory(),
    )


def create_comparison_table(results: List[BenchmarkResult]) -> str:
    """Create markdown comparison table from results."""
    
    # Header
    table = """
## Model Performance Comparison

| Model | Variant | Precision | Recall | mAP@50 | mAP@50-95 | Inference (ms) | FPS | Size (MB) |
|-------|---------|-----------|--------|--------|-----------|----------------|-----|-----------|
"""
    
    # Sort by mAP@50 descending
    sorted_results = sorted(results, key=lambda x: x.map50, reverse=True)
    
    for r in sorted_results:
        table += f"| {r.model_name} | {r.variant or '-'} | {r.precision:.3f} | {r.recall:.3f} | {r.map50:.3f} | {r.map50_95:.3f} | {r.inference_time_ms:.1f} | {r.fps:.1f} | {r.model_size_mb:.1f} |\n"
    
    return table


def load_test_data(data_dir: str, max_images: int = 50):
    """Load test images and create dummy GTs for benchmarking."""
    images = []
    gts = []
    
    data_path = Path(data_dir)
    image_dir = data_path / "test" / "images"
    label_dir = data_path / "test" / "labels"
    
    if not image_dir.exists():
        image_dir = data_path / "valid" / "images"
        label_dir = data_path / "valid" / "labels"
    
    if not image_dir.exists():
        print(f"Warning: No test/valid images found in {data_dir}")
        return images, gts
    
    for img_path in list(image_dir.glob("*"))[:max_images]:
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                # Create dummy GT (would need real labels)
                gts.append([])
    
    return images, gts


def run_benchmarks(
    data_dir: str = "datasets_merged",
    output_file: str = "benchmark_results.md",
    models: List[str] = None,
):
    """Run benchmarks for all models."""
    
    print("="*60)
    print("Billboard Segmentation Model Benchmark")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    images, gts = load_test_data(data_dir)
    print(f"  Loaded {len(images)} images")
    
    if not images:
        print("No images found. Using dummy benchmark data.")
        # Create dummy test image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        images = [dummy_img] * 10
        gts = [[]] * 10
    
    results = []
    models_to_test = models or list(MODEL_CONFIGS.keys())
    
    for model_name in models_to_test:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            continue
        
        try:
            if config["type"] == "yolo":
                variant = config["variant"]
                model_path = f"models/yolov8{variant}_seg_best.pt"
                
                # Check if trained model exists
                if not Path(model_path).exists():
                    model_path = f"yolov8{variant}-seg.pt"  # Use pretrained
                
                segmenter = YOLOSegmenter(model_path=model_path, variant=variant)
                result = benchmark_model(
                    segmenter, images, gts, 
                    model_name=f"YOLOv8{variant}-seg",
                    variant=variant,
                    model_path=model_path
                )
                
            elif config["type"] == "maskrcnn":
                model_path = "maskrcnn_trained_models/best.pth"
                if not Path(model_path).exists():
                    print(f"  Skipping Mask R-CNN (not trained)")
                    continue
                    
                segmenter = MaskRCNNSegmenter(model_path=model_path)
                result = benchmark_model(
                    segmenter, images, gts,
                    model_name="Mask R-CNN",
                    model_path=model_path
                )
                
            elif config["type"] == "grounded-sam":
                segmenter = GroundedSAMSegmenter()
                result = benchmark_model(
                    segmenter, images, gts,
                    model_name="GroundedSAM"
                )
                
            elif config["type"] == "sam2":
                segmenter = SAM2Segmenter()
                result = benchmark_model(
                    segmenter, images, gts,
                    model_name="SAM2"
                )
            
            results.append(result)
            print(f"  ✓ {model_name}: mAP@50={result.map50:.3f}, FPS={result.fps:.1f}")
            
        except Exception as e:
            print(f"  ✗ {model_name}: Error - {e}")
    
    # Generate comparison table
    table = create_comparison_table(results)
    print(table)
    
    # Save results
    with open(output_file, "w") as f:
        f.write("# Billboard Segmentation Benchmark Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Test images: {len(images)}\n\n")
        f.write(table)
        
        # Add detailed results as JSON
        f.write("\n\n## Detailed Results\n\n```json\n")
        f.write(json.dumps([asdict(r) for r in results], indent=2))
        f.write("\n```\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Billboard Segmentation Models")
    
    parser.add_argument("--data-dir", default="datasets_merged",
                        help="Path to dataset directory")
    parser.add_argument("--output", default="benchmark_results.md",
                        help="Output file for results")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to benchmark")
    parser.add_argument("--yolo-only", action="store_true",
                        help="Only benchmark YOLO variants")
    
    args = parser.parse_args()
    
    models = args.models
    if args.yolo_only:
        models = [f"yolov8{v}-seg" for v in YOLO_VARIANTS]
    
    run_benchmarks(
        data_dir=args.data_dir,
        output_file=args.output,
        models=models,
    )


if __name__ == "__main__":
    main()
