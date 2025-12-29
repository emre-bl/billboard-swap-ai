"""
Merge Billboard Datasets from Multiple Sources
Combines all downloaded Roboflow datasets into a unified training set.
Normalizes class names to single 'billboard' class (class_id=0).
"""
import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# Dataset configurations: defines which source classes to keep as billboard
DATASET_CONFIGS = {
    "billboard-xlvz1": {
        "keep_classes": {"billboard": 0},  # class_name: original_class_id
        "is_segmentation": False,  # Object detection only
    },
    "roadeye": {
        "keep_classes": {"billboard": 0},
        "is_segmentation": False,
    },
    "billboards-4zz9y": {
        "keep_classes": {"billboard": 0},
        "is_segmentation": False,
    },
    "dataset-billboard-video-2": {
        "keep_classes": {"Billboard": 0},  # Note capital B
        "is_segmentation": True,
    },
    "dataset-billboard-video-3": {
        "keep_classes": {"Billboard": 0},
        "is_segmentation": True,
    },
    "sports-videos": {
        "keep_classes": {"billboard": 4},  # billboard is class 4 in this dataset
        "is_segmentation": True,
    },
}

def load_data_yaml(dataset_path):
    """Load data.yaml from dataset and return class names mapping"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def convert_label_file(src_label_path, dst_label_path, keep_class_ids, is_segmentation):
    """
    Convert label file: keep only specified classes and remap to class 0.
    For object detection, convert bounding boxes to rough polygon masks.
    """
    if not os.path.exists(src_label_path):
        return False
    
    with open(src_label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        if class_id not in keep_class_ids:
            continue
        
        # All billboard classes become class 0
        parts[0] = "0"
        
        if not is_segmentation and len(parts) == 5:
            # Convert bbox to polygon: x_center, y_center, width, height -> 4 corner points
            x_c, y_c, w, h = map(float, parts[1:5])
            x1, y1 = x_c - w/2, y_c - h/2
            x2, y2 = x_c + w/2, y_c + h/2
            # Polygon format: x1 y1 x2 y1 x2 y2 x1 y2
            new_line = f"0 {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
            new_lines.append(new_line)
        else:
            # Already segmentation format, just remap class
            new_lines.append(" ".join(parts))
    
    if new_lines:
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
        with open(dst_label_path, 'w') as f:
            f.write("\n".join(new_lines))
        return True
    return False

def merge_datasets(datasets_dir="datasets", output_dir="datasets_merged", include_existing=True):
    """Merge all datasets into unified structure"""
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(lambda: {"images": 0, "labels": 0, "skipped": 0})
    
    # Process each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        dataset_path = datasets_dir / dataset_name
        if not dataset_path.exists():
            print(f"Skipping {dataset_name}: not found")
            continue
        
        print(f"\nProcessing: {dataset_name}")
        
        # Load class names from data.yaml
        data_yaml = load_data_yaml(dataset_path)
        if data_yaml is None:
            print(f"  Warning: No data.yaml found, skipping")
            continue
        
        class_names = data_yaml.get("names", [])
        print(f"  Classes: {class_names}")
        
        # Map class names to IDs that we want to keep
        keep_class_ids = set()
        for class_name, orig_id in config["keep_classes"].items():
            if isinstance(class_names, list):
                for idx, name in enumerate(class_names):
                    if name.lower() == class_name.lower():
                        keep_class_ids.add(idx)
                        break
            elif orig_id is not None:
                keep_class_ids.add(orig_id)
        
        print(f"  Keeping class IDs: {keep_class_ids}")
        
        # Process each split
        for split in ["train", "valid", "test"]:
            src_images = dataset_path / split / "images"
            src_labels = dataset_path / split / "labels"
            
            if not src_images.exists():
                continue
            
            for img_file in src_images.iterdir():
                if not img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                
                # Generate unique filename with dataset prefix
                new_name = f"{dataset_name}_{img_file.name}"
                dst_image = output_dir / split / "images" / new_name
                dst_label = output_dir / split / "labels" / (Path(new_name).stem + ".txt")
                
                # Get source label file
                label_name = img_file.stem + ".txt"
                src_label = src_labels / label_name
                
                # Convert label file (only keep billboard class)
                if convert_label_file(src_label, dst_label, keep_class_ids, config["is_segmentation"]):
                    # Copy image only if we have valid labels
                    shutil.copy2(img_file, dst_image)
                    stats[dataset_name]["images"] += 1
                    stats[dataset_name]["labels"] += 1
                else:
                    stats[dataset_name]["skipped"] += 1
    
    # Include existing project dataset if requested
    if include_existing:
        print("\nProcessing: existing project data")
        for split_map in [("train", "train"), ("valid", "valid")]:
            src_split, dst_split = split_map
            src_images = Path("train" if src_split == "train" else "valid") / "images"
            src_labels = Path("train" if src_split == "train" else "valid") / "labels"
            
            if not src_images.exists():
                continue
            
            for img_file in src_images.iterdir():
                if not img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                
                new_name = f"original_{img_file.name}"
                dst_image = output_dir / dst_split / "images" / new_name
                dst_label = output_dir / dst_split / "labels" / (Path(new_name).stem + ".txt")
                
                src_label = src_labels / (img_file.stem + ".txt")
                if src_label.exists():
                    shutil.copy2(img_file, dst_image)
                    shutil.copy2(src_label, dst_label)
                    stats["original"]["images"] += 1
                    stats["original"]["labels"] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    total_images = 0
    for name, s in stats.items():
        print(f"  {name}: {s['images']} images, {s['skipped']} skipped")
        total_images += s["images"]
    
    print(f"\nTotal: {total_images} images")
    
    # Count per split
    for split in ["train", "valid", "test"]:
        count = len(list((output_dir / split / "images").glob("*")))
        print(f"  {split}: {count} images")
    
    # Create unified data.yaml
    data_yaml_content = {
        "train": str(output_dir / "train" / "images"),
        "val": str(output_dir / "valid" / "images"),
        "test": str(output_dir / "test" / "images"),
        "nc": 1,
        "names": ["billboard"],
    }
    
    yaml_path = "data_merged.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print(f"\nCreated: {yaml_path}")
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge Roboflow Billboard Datasets")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory with downloaded datasets")
    parser.add_argument("--output-dir", default="datasets_merged", help="Output directory for merged dataset")
    parser.add_argument("--no-existing", action="store_true", help="Don't include existing project data")
    
    args = parser.parse_args()
    
    merge_datasets(args.datasets_dir, args.output_dir, include_existing=not args.no_existing)

if __name__ == "__main__":
    main()
