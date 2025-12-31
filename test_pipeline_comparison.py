import os
import subprocess
from pathlib import Path

def run_pipeline_comparison():
    # Use Hacettepe image as replacement
    replacement = "test_videos/hacettepe.png"
    videos = ['test_1.mp4', 'test_2.mp4', 'test_3.mp4']

    combinations = [
        # ======================
        # SAM2 Tracker (Reference)
        # ======================
        # 1. YOLOv8m (Detect) + SAM2 (Segment) + SAM2 (Track)
        {
            "name": "YOLOv8m_SAM2_SAM2Track",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "sam2"
        },
        
        # ======================
        # Kalman Filter Tracker
        # ======================
        # 2. YOLOv8m (Detect) + SAM2 (Segment) + Kalman (Track)
        {
            "name": "YOLOv8m_SAM2_KalmanTrack",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "kalman"
        },
        # 3. Mask R-CNN Standalone + Kalman (Track)
        {
            "name": "MaskRCNN_Standalone_KalmanTrack",
            "det_model": "none",  # Skip external detection, MaskRCNN handles both
            "det_path": "",
            "seg_model": "maskrcnn",
            "seg_path": "maskrcnn_trained_models/best.pth",
            "tracker": "kalman"
        },

        # ======================
        # Optical Flow Tracker
        # ======================
        # 4. YOLOv8m (Detect) + SAM2 (Segment) + OpticalFlow (Track)
        {
            "name": "YOLOv8m_SAM2_OpticalFlow",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "optical-flow"
        },

        # ======================
        # Homography / Feature Tracker
        # ======================
        # 5. YOLOv8m (Detect) + SAM2 (Segment) + FeatureHomography (SIFT/SuperPoint)
        {
            "name": "YOLOv8m_SAM2_FeatureHomo",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "feature-homography"
        },
        
        # ======================
        # ADVANCED TRACKERS (2025 State-of-the-Art)
        # ======================
        
        # 6. YOLOv8m (Detect) + SAM2 (Segment) + ECC Homography (Sub-pixel refinement)
        {
            "name": "YOLOv8m_SAM2_ECCHomography",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "ecc-homography"
        },
        
        # 7. YOLOv8m (Detect) + SAM2 (Segment) + Hybrid Flow (GFTT + FB error)
        {
            "name": "YOLOv8m_SAM2_HybridFlow",
            "det_model": "yolo",
            "det_path": "models/yolov8m_det_best.pt",
            "seg_model": "sam2",
            "seg_path": "models/sam2_b.pt",
            "tracker": "hybrid-flow"
        }
    ]
    
    output_base = Path("comparison_outputs")
    output_base.mkdir(exist_ok=True)
    
    print("Starting Pipeline Comparison Benchmark...")
    print("Testing SAM2, Kalman, and Optical Flow trackers.")
    print("="*60)
    
    for combo in combinations:
        print(f"\n--- Running {combo['name']} ---")
        save_dir = output_base / combo["name"]
        save_dir.mkdir(exist_ok=True)
        
        for video in videos:
            video_path = Path("test_videos") / video
            if not video_path.exists():
                print(f"Video {video} not found.")
                continue
            
            output_file = save_dir / video
            
            cmd = [
                "python", "src/scripts/process_video.py",
                "--video", str(video_path),
                "--replacement", replacement,
                "--output", str(output_file),
                "--det-model", combo["det_model"],
                "--det-path", combo["det_path"],
                "--seg-model", combo["seg_model"],
                "--seg-path", combo["seg_path"],
                "--tracker", combo["tracker"],
                "--conf", "0.7"
            ]
            
            print(f"Command: {' '.join(cmd)}")
            try:
                # Set PYTHONPATH to include src folder for module imports
                env = os.environ.copy()
                env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
                subprocess.run(cmd, check=True, env=env)
                print(f"✓ Completed: {combo['name']}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {e}")
    
    print("\n" + "="*60)
    print("Comparison complete! Check comparison_outputs/ for results.")

if __name__ == "__main__":
    run_pipeline_comparison()
