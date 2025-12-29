import os
import glob

def remap_labels(dataset_path, mapping, keep_others=False):
    """
    Remaps class IDs in label files.
    mapping: dict of {old_id: new_id}
    keep_others: if False, deletes lines with classes not in mapping
    """
    print(f"Processing {dataset_path}...")
    
    # Process train, valid, test
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        print(f"  - {split}: remapping labels...")
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                cls_id = int(parts[0])
                
                if cls_id in mapping:
                    new_cls = mapping[cls_id]
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts) + "\n")
                elif keep_others:
                    new_lines.append(line)
            
            # Rewrite file
            with open(label_file, 'w') as f:
                f.writelines(new_lines)

# --- CONFIG ---

# 1. Roadeye (Detection): 0->0, 1->0
remap_labels(
    "datasets/detection_datasets/roadeye",
    {0: 0, 1: 0},
    keep_others=False
)

# 2. Dataset-Billboard-Video-2 (Segmentation): 0->0, 1->0 (Video-art)
remap_labels(
    "datasets/segmentation_datasets/dataset-billboard-video-2",
    {0: 0, 1: 0}, 
    keep_others=False # Drop 'no-billboard' (2)
)

# 3. Dataset-Billboard-Video-3 (Segmentation): 0->0 (Video-art), 1->0 (Billboard)
remap_labels(
    "datasets/segmentation_datasets/dataset-billboard-video-3",
    {0: 0, 1: 0},
    keep_others=False
)

# 4. Sports-Videos (Segmentation): 1->0 (Video-Art), 3->0 (Billboard)
remap_labels(
    "datasets/segmentation_datasets/sports-videos",
    {1: 0, 3: 0},
    keep_others=False # Drop others
)
