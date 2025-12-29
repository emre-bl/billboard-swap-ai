"""
Download Billboard Datasets from Roboflow Universe
Downloads 7 datasets: 4 object detection + 3 instance segmentation
"""
import os
import argparse
from roboflow import Roboflow

# Dataset configurations: (workspace, project, version, format)
DATASETS = {
    # Object Detection Datasets
    "billboard-xlvz1": ("arslan-ongr8", "billboard-xlvz1", 1, "yolov8"),
    "billboard-detection-uo2ld": ("test-c8wix", "billboard-detection-uo2ld", 1, "yolov8"),
    "roadeye": ("roadeye", "roadeye", 2, "yolov8"),
    "billboards-4zz9y": ("image-processing-awivd", "billboards-4zz9y", 1, "yolov8"),
    
    # Instance Segmentation Datasets
    "dataset-billboard-video-2": ("cs231n-pp", "dataset-billboard-video-2", 1, "yolov8"),
    "dataset-billboard-video-3": ("cs231n-pp", "dataset-billboard-video-3", 6, "yolov8"),
    "sports-videos": ("cs231n-pp", "sports-videos", 2, "yolov8"),
}

def download_datasets(api_key: str, output_dir: str = "datasets", datasets_to_download: list = None):
    """Download specified datasets from Roboflow"""
    rf = Roboflow(api_key=api_key)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if datasets_to_download is None:
        datasets_to_download = list(DATASETS.keys())
    
    results = {}
    
    for name in datasets_to_download:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue
            
        workspace, project, version, format_type = DATASETS[name]
        dataset_dir = os.path.join(output_dir, name)
        
        print(f"\n{'='*50}")
        print(f"Downloading: {name}")
        print(f"  Workspace: {workspace}")
        print(f"  Project: {project}")
        print(f"  Version: {version}")
        print(f"  Format: {format_type}")
        print(f"{'='*50}")
        
        try:
            project_ref = rf.workspace(workspace).project(project)
            dataset = project_ref.version(version).download(format_type, location=dataset_dir)
            results[name] = {"status": "success", "path": dataset_dir}
            print(f"✓ Downloaded to: {dataset_dir}")
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            print(f"✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    success = sum(1 for r in results.values() if r["status"] == "success")
    print(f"Successful: {success}/{len(results)}")
    for name, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} {name}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Download Roboflow Billboard Datasets")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--output-dir", default="datasets", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=None, 
                        help="Specific datasets to download (default: all)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for name, (ws, proj, ver, fmt) in DATASETS.items():
            print(f"  - {name} ({ws}/{proj} v{ver})")
        return
    
    download_datasets(args.api_key, args.output_dir, args.datasets)

if __name__ == "__main__":
    main()
