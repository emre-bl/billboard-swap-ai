"""
Tracking Algorithm Benchmark System.

Benchmarks different tracking algorithms using video files.
Uses YOLOv8m for detection and SAM2 for segmentation at keyframes,
then propagates masks using various trackers while measuring accuracy
against keyframe-based pseudo ground truth.

Metrics:
- Frame IoU: IoU between predicted and keyframe-detected mask
- Mean IoU: Average IoU across sequence
- Success Rate @ IoU: Percentage of frames with IoU > threshold
- Track Failures: Frames where IoU drops below threshold
- Processing FPS: Frames per second
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.evaluation import calculate_iou
from modules.tracking import (
    BaseTracker,
    SAM2Tracker,
    OpticalFlowTracker,
    PlanarKalmanTracker,
    ByteTracker,
    FeatureHomographyTracker,
    ECCHomographyTracker,
    SAM2MemoryTracker,
    HybridFlowTracker,
    FusionTracker,
    AdaptiveOpticalFlowTracker,
)
from modules.detection import YOLODetector
from modules.segmentation import SAM2Segmenter


@dataclass
class TrackingMetrics:
    """Container for tracking evaluation metrics."""
    tracker_name: str
    video_name: str
    mean_iou: float
    median_iou: float
    std_iou: float
    success_rate_50: float  # % frames with IoU > 0.5
    success_rate_70: float  # % frames with IoU > 0.7
    track_failures: int      # frames with IoU < 0.1
    total_frames: int
    avg_fps: float
    frame_ious: List[float]  # Per-frame IoU for plotting


class VideoLoader:
    """Load video frames for benchmarking."""
    
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[Video] {self.video_path.name}: {self.total_frames} frames @ {self.fps:.1f} FPS ({self.width}x{self.height})")
    
    def load_frames(self, max_frames: Optional[int] = None, skip_frames: int = 0) -> List[np.ndarray]:
        """
        Load frames from video.
        
        Args:
            max_frames: Maximum number of frames to load (None = all)
            skip_frames: Skip every N frames (0 = no skip)
            
        Returns:
            List of BGR frames
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if skip_frames == 0 or frame_idx % (skip_frames + 1) == 0:
                frames.append(frame)
            
            frame_idx += 1
            
            if max_frames and len(frames) >= max_frames:
                break
        
        return frames
    
    def release(self):
        self.cap.release()


class TrackerBenchmark:
    """Benchmark a tracker on video sequence."""
    
    def __init__(self, detector_path: str, segmenter_path: str):
        """
        Initialize with detector and segmenter.
        
        Args:
            detector_path: Path to YOLOv8 detection model
            segmenter_path: Path to SAM2 segmentation model
        """
        self.detector_path = detector_path
        self.segmenter_path = segmenter_path
        
        # Initialize detector
        print("[Benchmark] Loading YOLOv8 detector...")
        self.detector = YOLODetector(detector_path)
        
        # Initialize segmenter
        print("[Benchmark] Loading SAM2 segmenter...")
        self.segmenter = SAM2Segmenter(segmenter_path)
    
    def detect_and_segment(self, frame: np.ndarray, conf: float = 0.5) -> Optional[np.ndarray]:
        """
        Detect and segment billboard in frame.
        
        Returns:
            Binary mask or None if no detection
        """
        try:
            detections = self.detector.detect(frame, conf=conf)
            if detections:
                # Use largest detection
                largest_det = max(detections, key=lambda d: 
                    (d[2] - d[0]) * (d[3] - d[1]))
                x1, y1, x2, y2 = largest_det[:4]
                
                # Segment using SAM2 - returns single mask or None
                mask = self.segmenter.segment_with_box(frame, [x1, y1, x2, y2])
                if mask is not None and np.sum(mask) > 0:
                    return mask
        except Exception as e:
            print(f"[Benchmark] Detection/segmentation failed: {e}")
        
        return None
    
    def run_video(
        self, 
        tracker: BaseTracker, 
        frames: List[np.ndarray],
        video_name: str,
        keyframe_interval: int = 30
    ) -> TrackingMetrics:
        """
        Run tracker on video and compute metrics.
        
        Uses detection+segmentation at keyframes as pseudo ground truth.
        
        Args:
            tracker: Tracker instance to evaluate
            frames: List of BGR frames
            video_name: Name of video for reporting
            keyframe_interval: Frames between keyframe re-detections
            
        Returns:
            TrackingMetrics with evaluation results
        """
        tracker_name = tracker.__class__.__name__
        frame_ious = []
        processing_times = []
        
        # Get initial mask from first frame
        print(f"  Detecting initial billboard...")
        init_mask = self.detect_and_segment(frames[0])
        
        if init_mask is None:
            print(f"  ERROR: Could not detect billboard in first frame")
            return TrackingMetrics(
                tracker_name=tracker_name,
                video_name=video_name,
                mean_iou=0.0,
                median_iou=0.0,
                std_iou=0.0,
                success_rate_50=0.0,
                success_rate_70=0.0,
                track_failures=len(frames),
                total_frames=len(frames),
                avg_fps=0.0,
                frame_ious=[0.0] * len(frames)
            )
        
        # Initialize tracker
        tracker.init(frames[0], init_mask)
        frame_ious.append(1.0)  # First frame is perfect by definition
        
        # Track through sequence
        current_gt_mask = init_mask
        
        for i in tqdm(range(1, len(frames)), desc=f"  Tracking", leave=False):
            start_time = time.perf_counter()
            
            # Track
            pred_mask = tracker.track(frames[i])
            
            elapsed = time.perf_counter() - start_time
            processing_times.append(elapsed)
            
            # At keyframes, update pseudo ground truth
            if i % keyframe_interval == 0:
                keyframe_mask = self.detect_and_segment(frames[i])
                if keyframe_mask is not None:
                    current_gt_mask = keyframe_mask
                    # Update tracker with new measurement if supported
                    if hasattr(tracker, 'update_measurement'):
                        tracker.update_measurement(keyframe_mask)
            
            # Compute IoU against current ground truth
            if pred_mask is not None:
                iou = calculate_iou(pred_mask, current_gt_mask)
            else:
                iou = 0.0
            
            frame_ious.append(iou)
        
        # Compute metrics
        frame_ious_arr = np.array(frame_ious)
        
        metrics = TrackingMetrics(
            tracker_name=tracker_name,
            video_name=video_name,
            mean_iou=float(np.mean(frame_ious_arr)),
            median_iou=float(np.median(frame_ious_arr)),
            std_iou=float(np.std(frame_ious_arr)),
            success_rate_50=float(np.mean(frame_ious_arr > 0.5) * 100),
            success_rate_70=float(np.mean(frame_ious_arr > 0.7) * 100),
            track_failures=int(np.sum(frame_ious_arr < 0.5)),
            total_frames=len(frames),
            avg_fps=1.0 / np.mean(processing_times) if processing_times else 0.0,
            frame_ious=frame_ious
        )
        
        return metrics


def create_all_trackers(sam2_path: str) -> List[Tuple[str, BaseTracker]]:
    """Create instances of all available trackers."""
    trackers = []
    
    # SAM2 Tracker
    try:
        trackers.append(("SAM2", SAM2Tracker(sam2_path)))
    except Exception as e:
        print(f"Failed to create SAM2Tracker: {e}")
    
    # Optical Flow
    try:
        trackers.append(("OpticalFlow", OpticalFlowTracker()))
    except Exception as e:
        print(f"Failed to create OpticalFlowTracker: {e}")
    
    # Kalman Filter
    try:
        trackers.append(("PlanarKalman", PlanarKalmanTracker()))
    except Exception as e:
        print(f"Failed to create PlanarKalmanTracker: {e}")
    
    # Feature Homography (SIFT)
    try:
        trackers.append(("FeatureHomography", FeatureHomographyTracker()))
    except Exception as e:
        print(f"Failed to create FeatureHomographyTracker: {e}")
    
    # ECC Homography
    try:
        trackers.append(("ECCHomography", ECCHomographyTracker()))
    except Exception as e:
        print(f"Failed to create ECCHomographyTracker: {e}")
    
    # Hybrid Flow
    try:
        trackers.append(("HybridFlow", HybridFlowTracker()))
    except Exception as e:
        print(f"Failed to create HybridFlowTracker: {e}")
    
    # Fusion Tracker (Multi-tracker fusion)
    try:
        trackers.append(("FusionTracker", FusionTracker(
            trackers=[
                ("optical-flow", AdaptiveOpticalFlowTracker()),
                ("feature-homography", FeatureHomographyTracker()),
            ],
            weights=[0.5, 0.5]
        )))
    except Exception as e:
        print(f"Failed to create FusionTracker: {e}")
    
    return trackers


def generate_markdown_table(results: List[TrackingMetrics], group_by_video: bool = False) -> str:
    """Generate markdown table from benchmark results."""
    
    if group_by_video:
        # Group results by video
        videos = {}
        for r in results:
            if r.video_name not in videos:
                videos[r.video_name] = []
            videos[r.video_name].append(r)
        
        lines = []
        for video_name, video_results in videos.items():
            lines.append(f"\n### {video_name}\n")
            video_results = sorted(video_results, key=lambda x: x.mean_iou, reverse=True)
            lines.append("| Tracker | Mean IoU | Median IoU | Success@0.5 | Success@0.7 | Failures | FPS |")
            lines.append("|---------|----------|------------|-------------|-------------|----------|-----|")
            
            for r in video_results:
                mean_str = f"**{r.mean_iou:.3f}**" if r == video_results[0] else f"{r.mean_iou:.3f}"
                lines.append(
                    f"| {r.tracker_name:18s} | {mean_str:8s} | {r.median_iou:.3f} | "
                    f"{r.success_rate_50:.1f}% | {r.success_rate_70:.1f}% | "
                    f"{r.track_failures:3d} | {r.avg_fps:.1f} |"
                )
        return "\n".join(lines)
    else:
        # Aggregate across videos
        tracker_stats = {}
        for r in results:
            if r.tracker_name not in tracker_stats:
                tracker_stats[r.tracker_name] = {
                    'ious': [], 'success50': [], 'success70': [], 
                    'failures': 0, 'total': 0, 'fps': []
                }
            tracker_stats[r.tracker_name]['ious'].extend(r.frame_ious)
            tracker_stats[r.tracker_name]['success50'].append(r.success_rate_50)
            tracker_stats[r.tracker_name]['success70'].append(r.success_rate_70)
            tracker_stats[r.tracker_name]['failures'] += r.track_failures
            tracker_stats[r.tracker_name]['total'] += r.total_frames
            tracker_stats[r.tracker_name]['fps'].append(r.avg_fps)
        
        # Create aggregated metrics
        agg_results = []
        for name, stats in tracker_stats.items():
            agg_results.append({
                'name': name,
                'mean_iou': np.mean(stats['ious']),
                'median_iou': np.median(stats['ious']),
                'success50': np.mean(stats['success50']),
                'success70': np.mean(stats['success70']),
                'failures': stats['failures'],
                'total': stats['total'],
                'fps': np.mean(stats['fps'])
            })
        
        agg_results = sorted(agg_results, key=lambda x: x['mean_iou'], reverse=True)
        
        lines = [
            "| Tracker | Mean IoU | Median IoU | Success@0.5 | Success@0.7 | Failures | FPS |",
            "|---------|----------|------------|-------------|-------------|----------|-----|"
        ]
        
        for i, r in enumerate(agg_results):
            mean_str = f"**{r['mean_iou']:.3f}**" if i == 0 else f"{r['mean_iou']:.3f}"
            lines.append(
                f"| {r['name']:18s} | {mean_str:8s} | {r['median_iou']:.3f} | "
                f"{r['success50']:.1f}% | {r['success70']:.1f}% | "
                f"{r['failures']:3d}/{r['total']} | {r['fps']:.1f} |"
            )
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark tracking algorithms on videos")
    parser.add_argument("--videos", type=str, nargs="+",
                       default=["test_videos/test_1.mp4", "test_videos/test_2.mp4", "test_videos/test_3.mp4"],
                       help="Path to video files")
    parser.add_argument("--video-dir", type=str, default=None,
                       help="Directory containing video files (alternative to --videos)")
    parser.add_argument("--det-model", type=str,
                       default="models/yolov8m_det_best.pt",
                       help="Path to YOLOv8 detection model")
    parser.add_argument("--seg-model", type=str,
                       default="sam2_b.pt",
                       help="Path to SAM2 segmentation model")
    parser.add_argument("--trackers", type=str, nargs="+",
                       default=["all"],
                       help="Trackers to benchmark (or 'all')")
    parser.add_argument("--keyframe-interval", type=int, default=30,
                       help="Frames between keyframe re-detections for pseudo ground truth")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process per video")
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="Skip every N frames (for faster testing)")
    parser.add_argument("--output", type=str,
                       default="benchmark_tracking_results.json",
                       help="Output JSON file")
    parser.add_argument("--group-by-video", action="store_true",
                       help="Show results grouped by video in output table")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tracking Algorithm Benchmark (Video Mode)")
    print("=" * 60)
    
    # Collect video files
    video_files = []
    if args.video_dir:
        video_dir = Path(args.video_dir)
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(video_dir.glob(f'*{ext}'))
        video_files = sorted(video_files)
    else:
        video_files = [Path(v) for v in args.videos]
    
    video_files = [v for v in video_files if v.exists()]
    
    if not video_files:
        print("No video files found!")
        return
    
    print(f"\nFound {len(video_files)} videos: {[v.name for v in video_files]}")
    
    # Initialize benchmark
    print("\n[1/2] Initializing benchmark...")
    benchmark = TrackerBenchmark(args.det_model, args.seg_model)
    
    # Create trackers
    print("\n[2/2] Running benchmarks...")
    if "all" in args.trackers:
        tracker_creators = {
            "SAM2": lambda: SAM2Tracker(args.seg_model),
            "OpticalFlow": lambda: OpticalFlowTracker(),
            "PlanarKalman": lambda: PlanarKalmanTracker(),
            "FeatureHomography": lambda: FeatureHomographyTracker(),
            "ECCHomography": lambda: ECCHomographyTracker(),
            "HybridFlow": lambda: HybridFlowTracker(),
        }
    else:
        tracker_map = {
            "sam2": lambda: SAM2Tracker(args.seg_model),
            "opticalflow": lambda: OpticalFlowTracker(),
            "planarkalman": lambda: PlanarKalmanTracker(),
            "kalman": lambda: PlanarKalmanTracker(),
            "featurehomography": lambda: FeatureHomographyTracker(),
            "ecchomography": lambda: ECCHomographyTracker(),
            "hybridflow": lambda: HybridFlowTracker(),
        }
        tracker_creators = {}
        for name in args.trackers:
            name_lower = name.lower().replace("-", "").replace("_", "")
            if name_lower in tracker_map:
                tracker_creators[name] = tracker_map[name_lower]
            else:
                print(f"Unknown tracker: {name}")
    
    if not tracker_creators:
        print("No trackers available!")
        return
    
    print(f"Benchmarking {len(tracker_creators)} trackers: {list(tracker_creators.keys())}")
    
    # Run benchmarks
    all_results = []
    
    for video_path in video_files:
        print(f"\n{'='*60}")
        print(f"Video: {video_path.name}")
        print("=" * 60)
        
        # Load video frames
        try:
            loader = VideoLoader(video_path)
            frames = loader.load_frames(max_frames=args.max_frames, skip_frames=args.skip_frames)
            loader.release()
        except Exception as e:
            print(f"Error loading video: {e}")
            continue
        
        if len(frames) < 10:
            print(f"Not enough frames ({len(frames)}), skipping")
            continue
        
        print(f"Loaded {len(frames)} frames")
        
        # Benchmark each tracker
        for tracker_name, tracker_creator in tracker_creators.items():
            print(f"\n--- {tracker_name} ---")
            try:
                # Create fresh tracker instance
                tracker = tracker_creator()
                
                metrics = benchmark.run_video(
                    tracker, frames, video_path.name,
                    keyframe_interval=args.keyframe_interval
                )
                all_results.append(metrics)
                
                print(f"  Mean IoU: {metrics.mean_iou:.3f}")
                print(f"  Success@0.5: {metrics.success_rate_50:.1f}%")
                print(f"  Failures: {metrics.track_failures}/{metrics.total_frames}")
                print(f"  FPS: {metrics.avg_fps:.1f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate outputs
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Markdown table
    table = generate_markdown_table(all_results, group_by_video=args.group_by_video)
    print("\n## Table 3: Tracking Algorithm Performance\n")
    print(table)
    
    # Save JSON
    output_data = {
        "videos": [str(v) for v in video_files],
        "keyframe_interval": args.keyframe_interval,
        "results": [asdict(r) for r in all_results]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
