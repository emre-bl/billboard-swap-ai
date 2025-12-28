"""
Tracking Module for Billboard Mask Propagation.

Supports:
- SAM2 propagation (built-in video tracking)
- Optical Flow (lightweight, fast)
- Cutie VOS (best accuracy, requires separate installation)
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional


class BaseTracker(ABC):
    """Abstract base class for video object tracking."""
    
    @abstractmethod
    def init(self, frame: np.ndarray, mask: np.ndarray):
        """
        Initialize tracker with first frame and mask.
        
        Args:
            frame: BGR image (H, W, 3)
            mask: Binary mask (H, W) with values 0 or 255
        """
        pass
    
    @abstractmethod
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Track object in new frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Tracked mask (H, W) or None if tracking failed
        """
        pass
    
    def reset(self):
        """Reset tracker state."""
        pass


class SAM2Tracker(BaseTracker):
    """
    SAM2 built-in video object segmentation.
    Uses SAM2's propagation capabilities for mask tracking.
    """
    
    def __init__(self, model_path: str = "sam2_b.pt"):
        from ultralytics import SAM
        self.model = SAM(model_path)
        self.prev_frame = None
        self.prev_mask = None
        self.initialized = False
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        self.initialized = True
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_mask is None:
            return None
        
        # Find bounding box from previous mask
        contours, _ = cv2.findContours(
            self.prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        
        bbox = [x, y, x + w, y + h]
        
        # Use SAM2 with bbox prompt
        results = self.model.predict(frame, bboxes=[bbox], verbose=False)
        
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
            tracked_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Update state
            self.prev_frame = frame.copy()
            self.prev_mask = tracked_mask
            
            return tracked_mask
        
        return None
    
    def reset(self):
        self.prev_frame = None
        self.prev_mask = None
        self.initialized = False


class OpticalFlowTracker(BaseTracker):
    """
    Optical flow-based tracking using Lucas-Kanade method.
    Fast but may drift over time.
    """
    
    def __init__(self):
        self.prev_gray = None
        self.prev_mask = None
        self.initialized = False
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_mask = mask.copy()
        self.initialized = True
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_gray is None:
            return None
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get contour points from previous mask
        contours, _ = cv2.findContours(
            self.prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Sample points along contour
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 4:
            return None
        
        p0 = approx.astype(np.float32)
        
        # Calculate optical flow
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, p0, None, **self.lk_params
        )
        
        if p1 is None:
            return None
        
        # Filter good points
        good_new = p1[status.flatten() == 1]
        
        if len(good_new) < 4:
            return None
        
        # Create new mask from tracked points
        new_mask = np.zeros_like(self.prev_mask)
        hull = cv2.convexHull(good_new.astype(np.int32))
        cv2.fillPoly(new_mask, [hull], 255)
        
        # Update state
        self.prev_gray = frame_gray.copy()
        self.prev_mask = new_mask
        
        return new_mask
    
    def reset(self):
        self.prev_gray = None
        self.prev_mask = None
        self.initialized = False


class CutieTracker(BaseTracker):
    """
    Cutie VOS (Video Object Segmentation) tracker.
    Best accuracy but requires Cutie installation.
    
    Note: Cutie needs to be installed separately:
    pip install cutie-video OR git clone https://github.com/hkchengrex/Cutie
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.initialized = False
        self._load_model()
    
    def _load_model(self):
        try:
            # Try importing Cutie
            from cutie.inference.inference_core import InferenceCore
            from cutie.utils.get_default_model import get_default_model
            
            self.model = get_default_model()
            self.processor = InferenceCore(self.model)
            print("Cutie VOS loaded successfully")
        except ImportError:
            print("Warning: Cutie not installed. Using SAM2 tracker as fallback.")
            self.model = None
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        if self.model is None:
            return
        
        # Convert mask to proper format
        mask_tensor = (mask > 127).astype(np.uint8)
        
        # Initialize Cutie with first frame and mask
        self.processor.step(frame, mask_tensor, idx_mask=False)
        self.initialized = True
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or not self.initialized:
            return None
        
        # Track using Cutie
        output = self.processor.step(frame)
        
        if output is not None:
            mask = (output > 0.5).astype(np.uint8) * 255
            return mask
        
        return None
    
    def reset(self):
        if self.processor is not None:
            self.processor.clear_memory()
        self.initialized = False


def create_tracker(config) -> BaseTracker:
    """
    Factory function to create tracker based on config.
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        BaseTracker instance
    """
    if config.tracker == "sam2":
        return SAM2Tracker(model_path=config.sam2_model_path)
    elif config.tracker == "optical-flow":
        return OpticalFlowTracker()
    elif config.tracker == "cutie":
        tracker = CutieTracker()
        # Fallback to SAM2 if Cutie not available
        if tracker.model is None:
            print("Falling back to SAM2 tracker")
            return SAM2Tracker(model_path=config.sam2_model_path)
        return tracker
    else:
        raise ValueError(f"Unknown tracker: {config.tracker}")
