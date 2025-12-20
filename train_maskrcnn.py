import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np
import os
import yaml

class BillboardDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transforms = transforms
        # Filter out images without annotations
        self.ids = []
        for img_id in sorted(self.coco.imgs.keys()):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.ids.append(img_id)
        print(f"Loaded {len(self.ids)} images with annotations")
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Get boxes and masks
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
            # Create mask from segmentation
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        
        # Ensure we have at least one valid annotation
        if len(boxes) == 0:
            # Create a dummy box (this shouldn't happen after filtering)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }
        
        return img, target
    
    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    # Load pre-trained Mask R-CNN
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        # Filter out empty targets
        valid_indices = [j for j, t in enumerate(targets) if len(t['boxes']) > 0]
        
        if len(valid_indices) == 0:
            continue
            
        images = [images[j].to(device) for j in valid_indices]
        targets = [{k: v.to(device) for k, v in targets[j].items()} for j in valid_indices]
        
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            valid_batches += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}/{len(data_loader)}, Loss: {losses.item():.4f}")
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    return total_loss / max(valid_batches, 1)

def main():
    # Load data.yaml
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = data_config['nc'] + 1  # +1 for background
    
    # Get image directories
    base_dir = os.path.dirname(os.path.abspath('data.yaml'))
    train_img_dir = os.path.join(base_dir, data_config['train'].replace('../', ''))
    val_img_dir = os.path.join(base_dir, data_config['val'].replace('../', ''))
    
    # Create datasets
    train_dataset = BillboardDataset('train_coco.json', train_img_dir)
    val_dataset = BillboardDataset('val_coco.json', val_img_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        lr_scheduler.step()
        
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'maskrcnn_trained_models/maskrcnn_billboard_epoch_{epoch + 1}.pth')
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()