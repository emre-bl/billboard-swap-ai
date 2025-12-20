import os
import json
import yaml
from PIL import Image
import numpy as np

def yolo_to_coco(data_yaml_path, output_json_path, split='train'):
    """Convert YOLO format to COCO format for Mask R-CNN"""
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get paths
    base_dir = os.path.dirname(data_yaml_path)
    images_dir = os.path.join(base_dir, data[split].replace('../', ''))
    labels_dir = images_dir.replace('images', 'labels')
    
    # Initialize COCO format
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, name in enumerate(data['names']):
        coco_format['categories'].append({
            "id": i + 1,
            "name": name,
            "supercategory": "none"
        })
    
    annotation_id = 1
    
    # Process each image
    for img_id, img_file in enumerate(sorted(os.listdir(images_dir)), 1):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        width, height = img.size
        
        # Add image info
        coco_format['images'].append({
            "id": img_id,
            "file_name": img_file,
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
                    segmentation = [[
                        x_min, y_min,
                        x_min + bbox_width, y_min,
                        x_min + bbox_width, y_min + bbox_height,
                        x_min, y_min + bbox_height
                    ]]
                    
                    coco_format['annotations'].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": segmentation,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Converted {split} split: {len(coco_format['images'])} images, {len(coco_format['annotations'])} annotations")

if __name__ == "__main__":
    # Convert all splits
    for split in ['train', 'val']:
        output_path = f'{split}_coco.json'
        yolo_to_coco('data.yaml', output_path, split)