import subprocess
from pathlib import Path
import os
import sys

# Append src to path for module imports if needed, though subprocess handles environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

videos = ["test_1.mp4", "test_2.mp4", "test_3.mp4", "test_4.mp4"]
replacement = "test_videos/hacettepe.png"
output_dir = "final_results"

Path(output_dir).mkdir(exist_ok=True)

# Base models path - assuming standard locations from previous context
# If these don't exist, the scripts handles defaults/fallbacks, but let's try to be explicit if possible.
# Ideally we trust the defaults in the scripts or the ones we found earlier.

det_path = "models/yolov8m_det_best.pt"
seg_path = "models/yolov8m_seg_best.pt"

# Env setup for pythonpath
env = os.environ.copy()
env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

for video in videos:
    video_path = f"test_videos/{video}"
    if not Path(video_path).exists():
        print(f"Skipping {video}, file not found.")
        continue

    print(f"\n{'='*20}\nProcessing {video}...\n{'='*20}")
    
    # ---------------------------------------------------------
    # 1. Run process_video.py (Standard V2 Pipeline)
    # ---------------------------------------------------------
    # We will use a standard robust configuration:
    # Detector: YOLOv8m
    # Segmenter: SAM2 (High quality)
    # Tracker: SAM2 (State of the art tracking)
    
    out_name_v2 = f"{Path(video).stem}_pipeline_v2_sam2.mp4"
    output_path_v2 = str(Path(output_dir) / out_name_v2)
    
    print(f"Running Standard Pipeline (v2) on {video} -> {output_path_v2}")
    
    cmd_v2 = [
        "python", "src/scripts/process_video.py",
        "--video", video_path,
        "--replacement", replacement,
        "--output", output_path_v2,
        "--det-model", "yolo",
        "--det-path", det_path, # Use trained model
        "--seg-model", "sam2",
        "--seg-path", "models/sam2_b.pt",
        "--tracker", "sam2",
        "--conf", "0.5"
    ]
    
    try:
        subprocess.run(cmd_v2, check=True, env=env)
        print("Standard Pipeline Completed.")
    except subprocess.CalledProcessError as e:
        print(f"Standard Pipeline Failed for {video}: {e}")

    # ---------------------------------------------------------
    # 2. Run process_video_comprehensive.py (All Approaches)
    # ---------------------------------------------------------
    # This runs: maskrcnn, yolo_sam, yolo_seg, grounded_sam
    # It generates multiple output files in the output dir.
    
    print(f"Running Comprehensive Test (All Modes) on {video}...")
    
    cmd_comp = [
        "python", "src/scripts/process_video_comprehensive.py",
        "--video", video_path,
        "--replacement", replacement,
        "--output-dir", output_dir, # Note: Script appends mode suffix
        "--mode", "all"
    ]
    
    try:
        subprocess.run(cmd_comp, check=True, env=env)
        print("Comprehensive Test Completed.")
    except subprocess.CalledProcessError as e:
        print(f"Comprehensive Test Failed for {video}: {e}")

print("\nBatch Processing Complete!")
