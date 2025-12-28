"""
Detection Module for Billboard Object Detection.

Supports:
- YOLOv8 (n, s, m, l, x variants)
- GroundingDINO (Zero-shot via Ultralytics YOLO World)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Literal

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> List[List[float]]:
        """
        Detect billboards in the frame.
        
        Args:
            frame: BGR image (H, W, 3)
            conf: Confidence threshold
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        pass
    
    def select_best(self, boxes: List[List[float]], method: str = "area") -> Optional[List[float]]:
        """Select best box based on area or confidence (if available)."""
        if not boxes:
            return None
        
        if method == "area":
            areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
            best_idx = np.argmax(areas)
            return boxes[best_idx]
        
        # Default to first if score not available in this simplified interface
        return boxes[0]

class YOLODetector(BaseDetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
    
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> List[List[float]]:
        results = self.model.predict(frame, conf=conf, verbose=False)
        boxes = []
        if results and results[0].boxes:
            # Filter by class if needed, but assuming model is trained on billboard or single class
            # xyxy format
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        return boxes

class GroundingDINODetector(BaseDetector):
    """Uses YOLO-World as a proxy for Zero-Shot detection (GroundingDINO equivalent in Ultralytics)."""
    def __init__(self, model_path: str = "yolov8s-world.pt", prompt: str = "billboard"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.set_classes([prompt])
    
    def detect(self, frame: np.ndarray, conf: float = 0.3) -> List[List[float]]:
        results = self.model.predict(frame, conf=conf, verbose=False)
        boxes = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        return boxes

def create_detector(model_type: str, model_path: str = None, prompt: str = "billboard") -> BaseDetector:
    if model_type == "yolo":
        path = model_path if model_path else "yolov8n.pt"
        return YOLODetector(path)
    elif model_type == "grounding-dino":
        path = model_path if model_path else "yolov8s-world.pt"
        return GroundingDINODetector(path, prompt)
    else:
        raise ValueError(f"Unknown detector type: {model_type}")
