"""
Unified Segmentation Module for Billboard Detection.

Supports:
- YOLOv8-seg (n, s, m, l, x variants) - Fine-tuned
- Mask R-CNN (ResNet50-FPN) - Fine-tuned  
- SAM2 - Zero-shot with auto-prompting
"""
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Literal
from pathlib import Path


class BaseSegmenter(ABC):
    """Abstract base class for all segmentation models."""
    
    @abstractmethod
    def segment(self, frame: np.ndarray, conf: float = 0.55) -> List[np.ndarray]:
        """
        Segment billboards in the frame.
        
        Args:
            frame: BGR image (H, W, 3)
            conf: Confidence threshold
            
        Returns:
            List of binary masks (H, W), each representing a detected billboard
        """
        pass
    
    def select_best(
        self, 
        masks: List[np.ndarray], 
        method: Literal["area", "confidence"] = "area"
    ) -> Optional[np.ndarray]:
        """
        Select the best billboard from multiple detections.
        
        Args:
            masks: List of binary masks
            method: Selection method - "area" for largest, "confidence" for highest score
            
        Returns:
            Best mask or None if no masks
        """
        if not masks:
            return None
        
        if method == "area":
            # Select mask with largest area
            areas = [np.sum(m > 0) for m in masks]
            best_idx = np.argmax(areas)
        else:
            # For models that provide confidence, this should be overridden
            best_idx = 0
            
        return masks[best_idx]


class YOLOSegmenter(BaseSegmenter):
    """YOLOv8-seg segmentation model."""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        variant: str = "n"
    ):
        from ultralytics import YOLO
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path).to(self.device)
        else:
            # Use pretrained model
            self.model = YOLO(f"yolov8{variant}-seg.pt").to(self.device)
        
        print(f"YOLO Segmenter initialized on {self.device}")
        self._confidences = []  # Store confidences for select_best
    
    def segment(self, frame: np.ndarray, conf: float = 0.55) -> List[np.ndarray]:
        results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
        
        masks = []
        self._confidences = []
        
        if results and results[0].masks is not None:
            mask_data = results[0].masks.data.cpu().numpy()
            
            # Get confidences if available
            if results[0].boxes is not None:
                self._confidences = results[0].boxes.conf.cpu().numpy().tolist()
            
            for mask in mask_data:
                # Resize mask to frame size if needed
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                # Convert to binary uint8
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                masks.append(binary_mask)
        
        return masks
    
    def select_best(
        self, 
        masks: List[np.ndarray], 
        method: Literal["area", "confidence"] = "area"
    ) -> Optional[np.ndarray]:
        if not masks:
            return None
        
        if method == "confidence" and self._confidences:
            best_idx = np.argmax(self._confidences)
        else:
            areas = [np.sum(m > 0) for m in masks]
            best_idx = np.argmax(areas)
            
        return masks[best_idx]


class MaskRCNNSegmenter(BaseSegmenter):
    """Mask R-CNN segmentation model (PyTorch)."""
    
    def __init__(self, model_path: str, num_classes: int = 2):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model architecture
        self.model = maskrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        # Load weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        self._scores = []
    
    def segment(self, frame: np.ndarray, conf: float = 0.55) -> List[np.ndarray]:
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        masks = []
        self._scores = []
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Filter by confidence
        scores = predictions['scores'].cpu().numpy()
        high_conf_indices = np.where(scores > conf)[0]
        
        if len(high_conf_indices) > 0:
            self._scores = scores[high_conf_indices].tolist()
            raw_masks = predictions['masks'][high_conf_indices].cpu().numpy()
            raw_masks = raw_masks.squeeze(1)  # Remove channel dim
            
            for mask in raw_masks:
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                masks.append(binary_mask)
        
        return masks
    
    def select_best(
        self, 
        masks: List[np.ndarray], 
        method: Literal["area", "confidence"] = "area"
    ) -> Optional[np.ndarray]:
        if not masks:
            return None
        
        if method == "confidence" and self._scores:
            best_idx = np.argmax(self._scores)
        else:
            areas = [np.sum(m > 0) for m in masks]
            best_idx = np.argmax(areas)
            
        return masks[best_idx]



class SAM2Segmenter(BaseSegmenter):
    """SAM2 segmentation model."""
    
    def __init__(self, model_path: str = "sam2_b.pt"):
        from ultralytics import SAM
        self.model = SAM(model_path)
        
    def segment(self, frame: np.ndarray, conf: float = 0.55) -> List[np.ndarray]:
        results = self.model(frame, verbose=False)
        masks = []
        if results and results[0].masks is not None:
            mask_data = results[0].masks.data.cpu().numpy()
            for mask in mask_data:
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                masks.append(binary_mask)
        return masks

    def segment_with_box(self, frame: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        """Segment with bounding box prompt."""
        # box: [x1, y1, x2, y2]
        results = self.model(frame, bboxes=[box], verbose=False)
        if results and results[0].masks is not None:
             mask_data = results[0].masks.data.cpu().numpy()
             if len(mask_data) > 0:
                 mask = mask_data[0]
                 if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                 return (mask > 0.5).astype(np.uint8) * 255
        return None

def create_segmenter(config) -> BaseSegmenter:
    """
    Factory function to create segmenter based on config.
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        BaseSegmenter instance
    """
    if config.segmenter == "yolov8-seg":
        return YOLOSegmenter(
            model_path=config.yolo_model_path,
            variant=config.yolo_variant
        )
    elif config.segmenter == "mask-rcnn":
        return MaskRCNNSegmenter(
            model_path=config.maskrcnn_model_path or "maskrcnn_trained_models/best.pth"
        )
    elif config.segmenter == "sam2":
        return SAM2Segmenter(model_path=config.sam2_model_path)
    else:
        raise ValueError(f"Unknown segmenter: {config.segmenter}")
