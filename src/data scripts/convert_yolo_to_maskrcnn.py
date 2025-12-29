import os
import json
import yaml
from PIL import Image
import numpy as np
import argparse

def yolo_to_coco(data_yaml_path, output_json_path, split='train'):
    """Convert YOLO format to COCO format for Mask R-CNN"""
    
    print(f"Converting {split} split from {data_yaml_path}...")
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get paths
    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    
    if split not in data:
        print(f"Split '{split}' not found in yaml.")
        return

    # Handle relative paths in yaml
    rel_paths = data[split]
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]
    
    # Initialize COCO format
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    if 'names' in data:
        names = data['names']
        # Handle dictionary or list format for names
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names.keys())]
            
        for i, name in enumerate(names):
            coco_format['categories'].append({
                "id": i + 1,
                "name": name,
                "supercategory": "none"
            })
    else:
        # Default category
        coco_format['categories'].append({"id": 1, "name": "billboard", "supercategory": "none"})
    
    annotation_id = 1
    image_id_counter = 1
    
    for rel_path in rel_paths:
        # data.yaml is usually in root, and paths might be 'datasets_merged/images/train'
        images_dir = os.path.join(base_dir, rel_path)
        images_dir = os.path.normpath(images_dir)
        
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found: {images_dir}")
            continue

        labels_dir = images_dir.replace('images', 'labels')
        
        # Process each image
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        try:
            files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])
        except Exception as e:
            print(f"Error listing files in {images_dir}: {e}")
            continue
            
        print(f"Found {len(files)} images in {images_dir}")
        
        for img_file in files:
            img_path = os.path.join(images_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue
            
            # Add image info
            coco_format['images'].append({
                "id": image_id_counter,
                "file_name": os.path.join(rel_path, img_file), # Use relative path to avoid name collisions if multiple datasets have same filenames
                "width": width,
                "height": height
            })
            
            # Read corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0]) + 1
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        
                        # Convert to COCO bbox format [x, y, width, height]
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2
                        
                        # Create segmentation (approximation as bbox polygon)
                        if len(parts) > 5:
                            # Polygon format: class x1 y1 x2 y2 ...
                            # YOLO polygon format is normalized
                            points = [float(x) for x in parts[1:]]
                            poly_points = []
                            for i in range(0, len(points), 2):
                                px = points[i] * width
                                py = points[i+1] * height
                                poly_points.append(px)
                                poly_points.append(py)
                            
                            segmentation = [poly_points]
                            
                            # Update bbox from polygon
                            pxs = poly_points[0::2]
                            pys = poly_points[1::2]
                            x_min = min(pxs)
                            y_min = min(pys)
                            bbox_width = max(pxs) - x_min
                            bbox_height = max(pys) - y_min
                            
                        else:
                            # Box format
                            segmentation = [[
                                x_min, y_min,
                                x_min + bbox_width, y_min,
                                x_min + bbox_width, y_min + bbox_height,
                                x_min, y_min + bbox_height
                            ]]
                        
                        coco_format['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id_counter,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "segmentation": segmentation,
                            "iscrowd": 0
                        })
                        annotation_id += 1
            
            image_id_counter += 1
    
    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Saved to {output_json_path}: {len(coco_format['images'])} images, {len(coco_format['annotations'])} annotations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format")
    parser.add_argument("--data", default="data_merged.yaml", help="Path to data.yaml file")
    args = parser.parse_args()

    # Convert all splits
    for split in ['train', 'val']:
        output_path = f'{split}_coco.json'
        yolo_to_coco(args.data, output_path, split)