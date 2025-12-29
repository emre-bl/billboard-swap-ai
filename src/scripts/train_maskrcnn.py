"""
Mask R-CNN Training Script for Billboard Segmentation.

Uses PyTorch and torchvision's Mask R-CNN with ResNet50-FPN backbone.
"""
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T


class BillboardDataset(Dataset):
    """Dataset for billboard segmentation in COCO format."""
    
    def __init__(self, images_dir: str, annotations_file: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = {}
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        # Filter images without annotations to prevent training crash
        self.image_ids = self._filter_invalid_images(coco_data)
        print(f"Loaded {len(self.image_ids)} images with annotations (filtered from {len(self.images)})")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        # Get annotations
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        masks = []
        labels = []
        
        for ann in anns:
            # Bounding box [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Segmentation mask
            if 'segmentation' in ann:
                mask = self._decode_segmentation(
                    ann['segmentation'], 
                    img_info['height'], 
                    img_info['width']
                )
                masks.append(mask)
            
            # Category (all billboards = class 1)
            labels.append(1)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
        }
        
        # Convert image to tensor
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        if self.transforms:
            img, target = self.transforms(img, target)
        if self.transforms:
            img, target = self.transforms(img, target)
            
        # Check if target is valid for RPN (must have boxes)
        if target['boxes'].shape[0] == 0:
            # Mask R-CNN training logic in torchvision generally expects at least one box per image
            # OR we need to handle negative samples correctly.
            # Standard generalized_rcnn.py asserts boxes.shape[0] > 0 during training in some versions or setup.
            # To be safe, we can try to skip this image or treat it as pure background if the model supports it.
            # However, the error `Expected target boxes to be a tensor of shape [N, 4], got torch.Size([0])`
            # confirms strict requirement.
            # Use specific invalid flag or filter in __init__.
            pass 

        return img, target

    def _filter_invalid_images(self, coco_data):
        valid_ids = []
        for img_id in self.image_ids:
            anns = self.annotations.get(img_id, [])
            if len(anns) > 0:
                valid_ids.append(img_id)
        return valid_ids
    
    def _decode_segmentation(self, segmentation, height, width):
        """Decode COCO segmentation to binary mask."""
        import pycocotools.mask as mask_utils
        
        if isinstance(segmentation, list):
            # Polygon format
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        else:
            # RLE format
            rle = segmentation
        
        mask = mask_utils.decode(rle)
        return mask


def get_model(num_classes: int = 2):
    """Create Mask R-CNN model with custom head."""
    
    # Load pretrained model
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


def collate_fn(batch):
    """Custom collate function for detection."""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"  Epoch {epoch} - Batch {num_batches}: Loss = {losses.item():.4f}")
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    # Simple evaluation - just compute loss
    total_loss = 0
    num_batches = 0
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Get losses in training mode briefly
        model.train()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        total_loss += losses.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_maskrcnn(
    train_images: str,
    train_annotations: str,
    val_images: str = None,
    val_annotations: str = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.005,
    output_dir: str = "maskrcnn_trained_models",
):
    """Train Mask R-CNN for billboard segmentation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create datasets
    train_dataset = BillboardDataset(train_images, train_annotations)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_images and val_annotations:
        val_dataset = BillboardDataset(val_images, val_annotations)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    # Create model
    model = get_model(num_classes=2)  # background + billboard
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3  # Default patience or passed as arg
    
    print(f"\n{'='*60}")
    print("Training Mask R-CNN for Billboard Segmentation")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Train images: {len(train_dataset)}")
    if val_loader:
        print(f"  Val images: {len(val_dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"  Train loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"  Val loss: {val_loss:.4f}")
            
            # Save best model and check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, output_path / 'best.pth')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ! Validation loss did not improve ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping.")
                    break
        
        lr_scheduler.step()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_path / f'epoch_{epoch}.pth')
    
    # Save final model state (even if stopped early, save the 'last' state)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_path / 'last.pth')
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {output_path / 'best.pth'}")
    print(f"  Last model: {output_path / 'last.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for Billboard Segmentation")
    
    parser.add_argument("--train-images", type=str, required=True,
                        help="Path to training images directory")
    parser.add_argument("--train-annotations", type=str, required=True,
                        help="Path to training annotations (COCO JSON)")
    parser.add_argument("--val-images", type=str, default=None,
                        help="Path to validation images directory")
    parser.add_argument("--val-annotations", type=str, default=None,
                        help="Path to validation annotations (COCO JSON)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="maskrcnn_trained_models",
                        help="Output directory for trained models")
    
    args = parser.parse_args()
    
    train_maskrcnn(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()