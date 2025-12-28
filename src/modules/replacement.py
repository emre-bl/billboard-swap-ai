"""
Replacement Module for Billboard Content Replacement.

Features:
- Perspective transformation
- Kalman filter for temporal smoothing
- Edge blending with Gaussian blur
- Mask refinement for occlusions
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points (4, 2)
        
    Returns:
        Ordered points (4, 2)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


class KalmanCornerSmoother:
    """
    Kalman filter for smoothing 4-corner coordinates.
    Reduces temporal jitter/flickering in video.
    """
    
    def __init__(
        self, 
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2
    ):
        # State: 8 values (x, y for each of 4 corners)
        self.state_dim = 8
        
        # Kalman filter matrices
        self.kf = cv2.KalmanFilter(self.state_dim, self.state_dim)
        
        # State transition matrix (identity - assume corners don't move much)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.eye(self.state_dim, dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(self.state_dim, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32)
        
        self.initialized = False
        self.prev_corners = None
    
    def update(self, corners: np.ndarray) -> np.ndarray:
        """
        Update Kalman filter with new corner measurements.
        
        Args:
            corners: 4 corner points (4, 2)
            
        Returns:
            Smoothed corners (4, 2)
        """
        measurement = corners.flatten().astype(np.float32)
        
        if not self.initialized:
            self.kf.statePost = measurement.reshape(-1, 1)
            self.initialized = True
            self.prev_corners = corners.copy()
            return corners
        
        # Check for large jumps (scene change or detection error)
        if self.prev_corners is not None:
            dist = np.linalg.norm(corners - self.prev_corners)
            if dist > 100:  # Reset if corners moved too much
                self.kf.statePost = measurement.reshape(-1, 1)
                self.prev_corners = corners.copy()
                return corners
        
        # Predict
        prediction = self.kf.predict()
        
        # Correct with measurement
        corrected = self.kf.correct(measurement.reshape(-1, 1))
        
        smoothed = corrected.flatten().reshape(4, 2)
        self.prev_corners = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        """Reset Kalman filter state."""
        self.initialized = False
        self.prev_corners = None
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)


class EMACornerSmoother:
    """
    Exponential Moving Average smoother for corners.
    Simpler alternative to Kalman filter.
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev_corners = None
    
    def update(self, corners: np.ndarray) -> np.ndarray:
        if self.prev_corners is None:
            self.prev_corners = corners.copy()
            return corners
        
        # Check for large jumps
        dist = np.linalg.norm(corners - self.prev_corners)
        if dist > 100:
            self.prev_corners = corners.copy()
            return corners
        
        # Exponential moving average
        smoothed = self.alpha * corners + (1 - self.alpha) * self.prev_corners
        self.prev_corners = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        self.prev_corners = None


class ReplacementEngine:
    """
    Engine for replacing billboard content with perspective transformation.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize smoother
        if config.smoothing_enabled:
            if config.smoothing_method == "kalman":
                self.smoother = KalmanCornerSmoother()
            else:
                self.smoother = EMACornerSmoother(alpha=config.ema_alpha)
        else:
            self.smoother = None
    
    def reset(self):
        """Reset engine state."""
        if self.smoother:
            self.smoother.reset()
    
    def extract_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 4 corners from binary mask.
        
        Args:
            mask: Binary mask (H, W) with values 0 or 255
            
        Returns:
            4 corner points (4, 2) or None
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fallback: use minimum area rectangle
            rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(rect).astype(np.float32)
        
        return order_points(corners)
    
    def blend_edges(
        self, 
        frame: np.ndarray, 
        warped: np.ndarray, 
        mask: np.ndarray,
        blur_radius: int = 5
    ) -> np.ndarray:
        """
        Blend replacement image edges with original frame.
        
        Args:
            frame: Original frame (H, W, 3)
            warped: Warped replacement image (H, W, 3)
            mask: Binary mask (H, W)
            blur_radius: Gaussian blur radius for edge blending
            
        Returns:
            Blended frame (H, W, 3)
        """
        # Create soft mask with blurred edges
        kernel_size = blur_radius * 2 + 1
        soft_mask = cv2.GaussianBlur(
            mask.astype(np.float32), 
            (kernel_size, kernel_size), 
            0
        )
        
        # Normalize to 0-1 range
        soft_mask = soft_mask / 255.0
        soft_mask = np.stack([soft_mask] * 3, axis=-1)
        
        # Alpha blend
        result = frame.astype(np.float32) * (1 - soft_mask) + warped.astype(np.float32) * soft_mask
        
        return result.astype(np.uint8)
    
    def apply(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray, 
        replacement_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply billboard replacement.
        
        Args:
            frame: Original frame (H, W, 3)
            mask: Binary mask (H, W)
            replacement_img: Replacement image (H, W, 3)
            
        Returns:
            Tuple of (result frame, corner points)
        """
        # Extract corners
        corners = self.extract_corners(mask)
        
        if corners is None:
            return frame, None
        
        # Apply smoothing
        if self.smoother:
            corners = self.smoother.update(corners)
        
        # Source points (replacement image corners)
        h, w = replacement_img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Destination points (billboard corners in frame)
        dst_pts = corners.astype(np.float32)
        
        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp replacement image
        warped = cv2.warpPerspective(
            replacement_img, 
            matrix, 
            (frame.shape[1], frame.shape[0])
        )
        
        # Create mask for warped region
        warped_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(warped_mask, [dst_pts.astype(np.int32)], 255)
        
        # Apply edge blending if enabled
        if self.config.edge_blend_enabled:
            result = self.blend_edges(
                frame, warped, warped_mask, 
                blur_radius=self.config.edge_blend_radius
            )
        else:
            # Simple overlay without blending
            result = frame.copy()
            mask_indices = warped_mask > 0
            result[mask_indices] = warped[mask_indices]
        
        return result, corners


# Backward compatibility
class CornerSmoother(EMACornerSmoother):
    """Backward compatible alias for EMACornerSmoother."""
    pass


def apply_warp(
    frame: np.ndarray, 
    mask: np.ndarray, 
    replacement_img: np.ndarray, 
    smoother: Optional[EMACornerSmoother] = None
) -> np.ndarray:
    """
    Backward compatible function for simple warp application.
    """
    # Create minimal config
    @dataclass
    class MinimalConfig:
        smoothing_enabled: bool = smoother is not None
        smoothing_method: str = "ema"
        ema_alpha: float = 0.5
        edge_blend_enabled: bool = False
        edge_blend_radius: int = 5
    
    engine = ReplacementEngine(MinimalConfig())
    if smoother:
        engine.smoother = smoother
    
    result, _ = engine.apply(frame, mask, replacement_img)
    return result
