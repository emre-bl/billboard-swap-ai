"""
Tracking Module for Billboard Mask Propagation.

Supports:
- SAM2 propagation (built-in video tracking)
- Optical Flow (lightweight, fast)
- Kalman Filter (lightweight, fast)
- Multi-tracker fusion for robustness
"""
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import deque
import warnings

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    warnings.warn("filterpy not installed. Kalman trackers will use simple fallback.")
    KalmanFilter = None


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
    
    def update_measurement(self, mask: np.ndarray):
        """Update tracker with external measurement (optional)."""
        pass
    
    def get_confidence(self) -> float:
        """Get tracking confidence score (0-1)."""
        return 0.5


@dataclass
class TrackingResult:
    mask: np.ndarray
    confidence: float
    tracker_id: str
    timestamp: float


class FusionTracker(BaseTracker):
    """
    Multi-tracker fusion for robust tracking.
    Combines results from multiple trackers using weighted voting.
    """
    
    def __init__(self, trackers: List[Tuple[str, BaseTracker]], 
                 weights: Optional[List[float]] = None):
        self.trackers = trackers
        self.weights = weights if weights else [1.0] * len(trackers)
        self.results_history = deque(maxlen=10)
        self.frame_shape = None
        self.initialized = False
        
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.frame_shape = frame.shape[:2]
        for _, tracker in self.trackers:
            tracker.init(frame, mask)
        self.initialized = True
        
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized:
            return None
            
        results = []
        valid_trackers = []
        
        # Get results from all trackers
        for tracker_id, tracker in self.trackers:
            try:
                mask = tracker.track(frame)
                if mask is not None:
                    confidence = tracker.get_confidence() if hasattr(tracker, 'get_confidence') else 0.7
                    results.append((mask, confidence, tracker_id))
                    valid_trackers.append((tracker, confidence))
            except Exception as e:
                print(f"[Fusion] Tracker {tracker_id} failed: {e}")
                continue
        
        if not results:
            return None
            
        # Weighted fusion
        fused_mask = self._fuse_masks(results)
        
        # Update tracker weights based on consistency
        self._update_weights(valid_trackers, fused_mask)
        
        return fused_mask
    
    def _fuse_masks(self, results: List[Tuple[np.ndarray, float, str]]) -> np.ndarray:
        """Fuse multiple masks using weighted voting."""
        h, w = self.frame_shape
        fused = np.zeros((h, w), dtype=np.float32)
        total_weight = 0.0
        
        for mask, confidence, _ in results:
            mask_float = mask.astype(np.float32) / 255.0
            fused += mask_float * confidence
            total_weight += confidence
        
        if total_weight > 0:
            fused = fused / total_weight
        
        # Threshold and convert to uint8
        fused_mask = (fused > 0.5).astype(np.uint8) * 255
        
        # Keep only largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fused_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fused_mask = np.uint8(labels == largest_label) * 255
        
        return fused_mask
    
    def _update_weights(self, trackers: List[Tuple[BaseTracker, float]], 
                       fused_mask: np.ndarray):
        """Update tracker weights based on agreement with fused result."""
        # Simple implementation: increase weight for agreeing trackers
        for tracker, old_weight in trackers:
            if hasattr(tracker, 'get_confidence'):
                mask = getattr(tracker, 'last_mask', None)
                if mask is not None:
                    # Compute agreement with fused mask
                    agreement = self._compute_mask_iou(mask, fused_mask)
                    # Update weight (slow adaptation)
                    new_weight = old_weight * 0.9 + agreement * 0.1
                    # Update tracker confidence if possible
                    if hasattr(tracker, 'tracking_confidence'):
                        tracker.tracking_confidence = min(1.0, tracker.tracking_confidence + 0.05)
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two masks."""
        mask1_bin = (mask1 > 127).astype(np.uint8)
        mask2_bin = (mask2 > 127).astype(np.uint8)
        
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def get_confidence(self) -> float:
        """Get overall fusion confidence."""
        if not self.trackers:
            return 0.0
        
        confidences = []
        for _, tracker in self.trackers:
            if hasattr(tracker, 'get_confidence'):
                confidences.append(tracker.get_confidence())
        
        return np.mean(confidences) if confidences else 0.5
    
    def reset(self):
        for _, tracker in self.trackers:
            tracker.reset()
        self.initialized = False


class SAM2Tracker(BaseTracker):
    """
    SAM2 tracker with improved prompting and error recovery.
    """
    
    def __init__(self, model_path: str = "sam2_b.pt", use_mask_prompt: bool = True):
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError("ultralytics not installed. Please install with 'pip install ultralytics'")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAM(model_path).to(self.device)
        self.use_mask_prompt = use_mask_prompt
        
        self.prev_frame = None
        self.prev_mask = None
        self.prev_bbox = None
        self.tracking_confidence = 1.0
        self.consecutive_failures = 0
        self.mask_history = deque(maxlen=5)
        self.initialized = False
        
        print(f"SAM2Tracker initialized on {self.device}")
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        self.prev_bbox = self._mask_to_bbox(mask)
        self.mask_history.append(mask.copy())
        self.tracking_confidence = 1.0
        self.consecutive_failures = 0
        self.initialized = True
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_mask is None:
            return None
        
        # Try multiple prompting strategies
        mask = self._track_with_bbox(frame)
        
        if mask is None and self.use_mask_prompt:
            mask = self._track_with_mask_prompt(frame)
        
        if mask is None:
            # Try motion prediction as last resort
            mask = self._predict_from_history(frame)
        
        # Validate and update
        if mask is not None and self._validate_mask(mask):
            self.consecutive_failures = 0
            self.tracking_confidence = min(1.0, self.tracking_confidence + 0.05)
            
            # Update state
            self.prev_frame = frame.copy()
            self.prev_mask = mask.copy()
            self.prev_bbox = self._mask_to_bbox(mask)
            self.mask_history.append(mask.copy())
        else:
            self.consecutive_failures += 1
            self.tracking_confidence = max(0.1, self.tracking_confidence * 0.7)
            
            if self.consecutive_failures > 3:
                print("[SAM2] Multiple consecutive failures, resetting")
                self.reset()
                return None
            
            # Use last valid mask
            mask = self.prev_mask
        
        return mask
    
    def _track_with_bbox(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Track using bounding box prompt."""
        if self.prev_bbox is None:
            return None
        
        # Expand bbox slightly
        x1, y1, x2, y2 = self.prev_bbox
        pad = 15
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        
        bbox = [x1, y1, x2, y2]
        
        try:
            results = self.model.predict(frame, bboxes=[bbox], device=self.device, verbose=False)
            if results and results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                mask = self._process_raw_mask(mask, frame.shape)
                return mask
        except Exception as e:
            print(f"[SAM2] BBox tracking failed: {e}")
        
        return None
    
    def _track_with_mask_prompt(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Track using mask as prompt (if supported)."""
        if self.prev_mask is None:
            return None
        
        # Some SAM2 versions support mask prompting
        # This is a fallback implementation
        try:
            # Use previous mask to get refined bbox
            contours, _ = cv2.findContours(self.prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Create multiple point prompts from mask
                ys, xs = np.where(self.prev_mask > 127)
                if len(xs) > 0:
                    # Sample points from mask
                    indices = np.random.choice(len(xs), min(10, len(xs)), replace=False)
                    points = np.column_stack([xs[indices], ys[indices]]).tolist()
                    
                    # Run SAM2 with point prompts
                    results = self.model.predict(
                        frame, 
                        points=[points],
                        device=self.device,
                        verbose=False
                    )
                    
                    if results and results[0].masks is not None:
                        mask = results[0].masks.data[0].cpu().numpy()
                        mask = self._process_raw_mask(mask, frame.shape)
                        return mask
        except Exception as e:
            print(f"[SAM2] Mask prompt tracking failed: {e}")
        
        return None
    
    def _predict_from_history(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Predict mask position based on motion history."""
        if len(self.mask_history) < 2:
            return self.prev_mask
        
        # Estimate motion using optical flow
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get feature points from previous mask
        points = self._get_feature_points(self.prev_mask, prev_gray)
        if points is None or len(points) < 4:
            return self.prev_mask
        
        # Calculate optical flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if p1 is None or st is None:
            return self.prev_mask
        
        # Estimate transformation
        good_old = points[st.flatten() == 1]
        good_new = p1[st.flatten() == 1]
        
        if len(good_old) >= 4:
            M, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            if M is not None:
                # Warp previous mask
                h, w = frame.shape[:2]
                predicted_mask = cv2.warpPerspective(self.prev_mask, M, (w, h))
                return predicted_mask
        
        return self.prev_mask
    
    def _get_feature_points(self, mask: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """Get good feature points inside mask."""
        mask_u8 = (mask > 127).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask_u8
        )
        return points
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[List[int]]:
        """Convert mask to bounding box."""
        if mask is None:
            return None
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None
        return [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))]
    
    def _process_raw_mask(self, raw_mask: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """Process raw SAM2 mask output."""
        if raw_mask.shape != frame_shape[:2]:
            raw_mask = cv2.resize(raw_mask.astype(np.float32), 
                                 (frame_shape[1], frame_shape[0]))
        
        # Threshold
        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _validate_mask(self, mask: np.ndarray) -> bool:
        """Validate tracked mask."""
        if mask is None:
            return False
        
        area = np.sum(mask > 127)
        if area < 100:  # Too small
            return False
        
        if self.prev_mask is not None:
            prev_area = np.sum(self.prev_mask > 127)
            if prev_area > 0:
                area_ratio = area / prev_area
                if area_ratio < 0.3 or area_ratio > 3.0:
                    return False
        
        return True
    
    def get_confidence(self) -> float:
        return self.tracking_confidence
    
    def reset(self):
        self.prev_frame = None
        self.prev_mask = None
        self.prev_bbox = None
        self.mask_history.clear()
        self.tracking_confidence = 1.0
        self.consecutive_failures = 0
        self.initialized = False


class AdaptiveOpticalFlowTracker(BaseTracker):
    """
    Adaptive Optical Flow tracker with pyramid levels and error recovery.
    """
    
    def __init__(self, 
                 pyramid_levels: int = 3,
                 min_features: int = 20,
                 recovery_enabled: bool = True):
        self.pyramid_levels = pyramid_levels
        self.min_features = min_features
        self.recovery_enabled = recovery_enabled
        
        self.prev_gray = None
        self.prev_mask = None
        self.prev_points = None
        self.initial_mask = None
        
        self.tracking_confidence = 1.0
        self.feature_history = deque(maxlen=10)
        self.initialized = False
        
        # LK parameters
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_mask = mask.copy()
        self.initial_mask = mask.copy()
        
        # Extract features from mask region
        self.prev_points = self._extract_features(self.prev_gray, mask)
        if self.prev_points is not None and len(self.prev_points) > 0:
            self.feature_history.append(len(self.prev_points))
        
        self.tracking_confidence = 1.0
        self.initialized = True
        
        print(f"AdaptiveOpticalFlowTracker initialized with {len(self.prev_points) if self.prev_points is not None else 0} features")
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_points is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Forward-backward error checking for robust tracking
        new_points, status, error = self._track_with_fb_error(gray)
        
        if new_points is None or np.sum(status) < self.min_features:
            self.tracking_confidence = max(0.1, self.tracking_confidence * 0.5)
            
            if self.recovery_enabled:
                return self._attempt_recovery(gray, frame)
            return None
        
        # Estimate transformation
        good_old = self.prev_points[status.flatten() == 1]
        good_new = new_points[status.flatten() == 1]
        
        if len(good_old) >= 4:
            # Try affine first (faster), fall back to homography
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
            
            if M is not None:
                # Convert affine to homography for warpPerspective
                M_homo = np.vstack([M, [0, 0, 1]])
                new_mask = cv2.warpPerspective(self.prev_mask, M_homo, 
                                             (frame.shape[1], frame.shape[0]))
            else:
                # Try full homography
                M, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                if M is not None:
                    new_mask = cv2.warpPerspective(self.prev_mask, M, 
                                                 (frame.shape[1], frame.shape[0]))
                else:
                    self.tracking_confidence *= 0.7
                    return self.prev_mask
            
            # Update features in new mask region
            self.prev_points = self._extract_features(gray, new_mask)
            if self.prev_points is not None:
                self.feature_history.append(len(self.prev_points))
            
            # Update state
            self.prev_gray = gray.copy()
            self.prev_mask = new_mask.copy()
            
            # Update confidence based on feature count and error
            avg_error = np.mean(error[status.flatten() == 1]) if error is not None else 0
            self.tracking_confidence = self._compute_confidence(
                len(good_new), avg_error
            )
            
            return new_mask
        
        self.tracking_confidence *= 0.6
        return self.prev_mask
    
    def _track_with_fb_error(self, gray: np.ndarray) -> Tuple:
        """Track with forward-backward error checking."""
        # Forward flow
        new_points_fwd, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if new_points_fwd is None:
            return None, None, None
        
        # Backward flow
        back_points, status_back, err_back = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, new_points_fwd, None, **self.lk_params
        )
        
        if back_points is None:
            return new_points_fwd, status_fwd, err_fwd
        
        # Compute forward-backward error
        fb_error = np.abs(self.prev_points - back_points).reshape(-1, 2)
        fb_error = np.linalg.norm(fb_error, axis=1)
        
        # Combine status
        status_combined = (status_fwd.flatten() == 1) & (status_back.flatten() == 1)
        status_combined = status_combined & (fb_error < 1.0)  # Threshold
        
        return new_points_fwd, status_combined.reshape(-1, 1), err_fwd
    
    def _extract_features(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract good features to track within mask."""
        mask_u8 = (mask > 127).astype(np.uint8)
        
        # Adaptive number of features based on mask size
        mask_area = np.sum(mask_u8)
        max_features = max(self.min_features, int(mask_area / 100))
        
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask_u8
        )
        
        return points
    
    def _attempt_recovery(self, gray: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to recover from tracking failure."""
        # Try to re-initialize with initial mask (global search)
        if self.initial_mask is not None:
            # Resample features from initial mask region
            self.prev_points = self._extract_features(gray, self.initial_mask)
            if self.prev_points is not None and len(self.prev_points) >= self.min_features:
                self.prev_gray = gray.copy()
                self.prev_mask = self.initial_mask.copy()
                self.tracking_confidence = 0.3
                return self.prev_mask
        
        return None
    
    def _compute_confidence(self, num_features: int, avg_error: float) -> float:
        """Compute tracking confidence based on features and error."""
        if num_features < self.min_features:
            return 0.2
        
        # Feature count confidence
        feature_conf = min(1.0, num_features / 100)
        
        # Error confidence (lower error is better)
        error_conf = max(0.1, 1.0 - min(1.0, avg_error / 10))
        
        # Combined confidence
        confidence = 0.7 * feature_conf + 0.3 * error_conf
        
        # Apply temporal smoothing
        self.tracking_confidence = 0.8 * self.tracking_confidence + 0.2 * confidence
        
        return self.tracking_confidence
    
    def get_confidence(self) -> float:
        return self.tracking_confidence
    
    def reset(self):
        self.prev_gray = None
        self.prev_mask = None
        self.prev_points = None
        self.feature_history.clear()
        self.tracking_confidence = 1.0
        self.initialized = False


# Alias for backward compatibility
OpticalFlowTracker = AdaptiveOpticalFlowTracker



class PlanarKalmanTracker(BaseTracker):
    """
    Kalman Filter tracking 8D corner coordinates with optical flow measurements.
    """
    
    def __init__(self, adaptive_noise: bool = True, outlier_threshold: float = 3.0):
        if KalmanFilter is None:
            raise ImportError("filterpy not installed. Run 'pip install filterpy'")
        
        self.kf = None
        self.prev_mask = None
        self.prev_frame = None
        self.prev_gray = None
        self.prev_corners = None
        self.initialized = False
        
        # Optical flow parameters for measurement
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Adaptive noise settings
        self.adaptive_noise = adaptive_noise
        self.outlier_threshold = outlier_threshold
        self.tracking_quality = 1.0
        
        # Base noise values
        self.base_R = 5.0
        self.base_Q = 0.1
    
    def _mask_to_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract 4 ordered corners from mask using improved algorithm."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Use Douglas-Peucker algorithm for better approximation
        epsilon = 0.015 * cv2.arcLength(cnt, True)  # Tighter epsilon
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Try to get exactly 4 corners
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        elif len(approx) > 4:
            # Use convex hull and find 4 extreme points
            hull = cv2.convexHull(cnt)
            corners = self._get_quad_corners(hull)
        else:
            # Fallback: use minimum area rectangle
            rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(rect)
        
        return self._order_corners(corners)
    
    def _get_quad_corners(self, hull: np.ndarray) -> np.ndarray:
        """Get 4 corners from convex hull using extreme points."""
        hull = hull.reshape(-1, 2)
        
        # Find extreme points in different directions
        sum_pts = hull.sum(axis=1)
        diff_pts = np.diff(hull, axis=1).flatten()
        
        tl = hull[np.argmin(sum_pts)]      # Top-left (min x+y)
        br = hull[np.argmax(sum_pts)]      # Bottom-right (max x+y)
        tr = hull[np.argmin(diff_pts)]     # Top-right (min y-x)
        bl = hull[np.argmax(diff_pts)]     # Bottom-left (max y-x)
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        corners = corners.reshape(4, 2)
        
        # Sort by y-coordinate
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        
        # Top two and bottom two
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        
        # Sort by x-coordinate
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def _corners_to_mask(self, corners: np.ndarray, shape: tuple) -> np.ndarray:
        """Create mask from 4 corners."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def _track_corners_optical_flow(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Track corners using optical flow for measurement."""
        if self.prev_gray is None or self.prev_corners is None:
            return None
        
        # Convert corners to the format expected by optical flow
        p0 = self.prev_corners.reshape(-1, 1, 2).astype(np.float32)
        
        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )
        
        if p1 is None or status is None:
            return None
        
        # Check if all corners were tracked
        if status.sum() < 4:
            return None
        
        # Calculate tracking quality based on error
        if error is not None:
            avg_error = error[status == 1].mean()
            self.tracking_quality = 1.0 / (1.0 + avg_error)
        
        return p1.reshape(4, 2)
    
    def _compute_mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """Compute Mahalanobis distance for outlier detection."""
        if self.kf is None:
            return 0.0
        
        # Innovation (measurement - predicted measurement)
        innovation = measurement.reshape(-1, 1) - self.kf.H @ self.kf.x
        
        # Innovation covariance
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        
        # Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            distance = np.sqrt(innovation.T @ S_inv @ innovation)
            return float(distance)
        except:
            return 0.0
    
    def _update_adaptive_noise(self):
        """Update process and measurement noise based on tracking quality."""
        if not self.adaptive_noise:
            return
        
        # Increase measurement noise when tracking quality is low
        noise_factor = 1.0 / max(self.tracking_quality, 0.1)
        self.kf.R = np.eye(8) * (self.base_R * noise_factor)
        
        # Increase process noise when tracking is uncertain
        self.kf.Q = np.eye(16) * (self.base_Q * noise_factor)
        self.kf.Q[8:, 8:] *= 0.5  # Less noise on velocities
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_mask = mask.copy()
        self.prev_frame = frame.copy()
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners = self._mask_to_corners(mask)
        
        if corners is None:
            print("PlanarKalman: Could not extract corners from mask.")
            return
        
        self.prev_corners = corners
        
        # State: 16D [x1, y1, x2, y2, x3, y3, x4, y4, vx1, vy1, ..., vx4, vy4]
        self.kf = KalmanFilter(dim_x=16, dim_z=8)
        
        # Transition matrix (constant velocity model)
        dt = 1.0
        self.kf.F = np.eye(16)
        for i in range(8):
            self.kf.F[i, i + 8] = dt
        
        # Measurement matrix (we measure positions only)
        self.kf.H = np.zeros((8, 16))
        for i in range(8):
            self.kf.H[i, i] = 1.0
        
        # Initial covariances
        self.kf.P *= 100.0
        self.kf.R = np.eye(8) * self.base_R
        self.kf.Q = np.eye(16) * self.base_Q
        self.kf.Q[8:, 8:] *= 0.5  # Less noise on velocities
        
        # Initial state
        state = np.zeros(16)
        state[:8] = corners.flatten()
        self.kf.x = state.reshape(-1, 1)
        
        self.frame_shape = frame.shape
        self.initialized = True
        self.tracking_quality = 1.0
        print(f"PlanarKalman initialized with 8D corner tracking (adaptive={self.adaptive_noise}).")
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.kf is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        
        # Predict step
        self.kf.predict()
        
        # Try to get measurement from optical flow
        measured_corners = self._track_corners_optical_flow(gray)
        
        if measured_corners is not None:
            # Clamp measurements to frame bounds
            measured_corners[:, 0] = np.clip(measured_corners[:, 0], 0, w - 1)
            measured_corners[:, 1] = np.clip(measured_corners[:, 1], 0, h - 1)
            
            # Outlier rejection using Mahalanobis distance
            mahal_dist = self._compute_mahalanobis_distance(measured_corners.flatten())
            
            if mahal_dist < self.outlier_threshold:
                # Update step with measurement
                self.kf.update(measured_corners.flatten().reshape(-1, 1))
                
                # Update adaptive noise
                self._update_adaptive_noise()
            else:
                # Measurement is an outlier, skip update
                self.tracking_quality *= 0.9
        else:
            # No measurement available, reduce tracking quality
            self.tracking_quality *= 0.8
        
        # Get filtered corners from state
        corners = self.kf.x[:8].flatten().reshape(4, 2)
        
        # Clamp to frame bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
        
        # Create mask
        new_mask = self._corners_to_mask(corners, frame.shape)
        
        # Update state for next iteration
        self.prev_mask = new_mask
        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()
        self.prev_corners = corners
        
        return new_mask
    
    def update_measurement(self, mask: np.ndarray):
        """Update Kalman filter with external measurement from detection/segmentation."""
        if not self.initialized or self.kf is None:
            return
        
        corners = self._mask_to_corners(mask)
        if corners is not None:
            # Check if measurement is reasonable
            mahal_dist = self._compute_mahalanobis_distance(corners.flatten())
            
            if mahal_dist < self.outlier_threshold * 2:  # More lenient for external measurements
                self.kf.update(corners.flatten().reshape(-1, 1))
                self.prev_corners = corners
                self.tracking_quality = min(1.0, self.tracking_quality + 0.1)
    
    def get_confidence(self) -> float:
        """Get current tracking quality metric (0-1)."""
        return self.tracking_quality
    
    def reset(self):
        self.kf = None
        self.prev_mask = None
        self.prev_frame = None
        self.prev_gray = None
        self.prev_corners = None
        self.initialized = False
        self.tracking_quality = 1.0


class FeatureHomographyTracker(BaseTracker):
    """
    Feature-based Homography Tracker (SIFT/SuperPoint).
    Computes Homography between frames to propagate mask.
    """
    
    def __init__(self, feature_type="sift"):
        self.feature_type = feature_type
        if feature_type == "sift":
            self.detector = cv2.SIFT_create()
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Fallback to ORB
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        self.prev_frame = None
        self.prev_mask = None
        self.prev_kp = None
        self.prev_des = None
        self.initialized = False
        
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        
        # Detect features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        
        if kp is None or len(kp) < 4:
            print("FeatureHomography: Not enough features in init.")
            return

        # Keep only features inside the mask
        height, width = mask.shape
        good_kp = []
        good_des = []
        
        for k, d in zip(kp, des):
            x, y = int(k.pt[0]), int(k.pt[1])
            if 0 <= x < width and 0 <= y < height:
                if mask[y, x] > 127:
                    good_kp.append(k)
                    good_des.append(d)
                    
        if len(good_kp) < 4:
            print("FeatureHomography: Not enough features inside mask.")
            return
            
        self.prev_kp = good_kp
        self.prev_des = np.array(good_des)
        self.initialized = True
        print(f"FeatureHomography initialized with {len(good_kp)} features.")

    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_des is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        
        if kp is None or len(kp) < 4:
            return None
            
        # Match
        if self.feature_type == "sift":
            matches = self.matcher.knnMatch(self.prev_des, des, k=2)
            # Ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:
            good_matches = self.matcher.match(self.prev_des, des)
            good_matches = sorted(good_matches, key = lambda x:x.distance)
            good_matches = good_matches[:int(len(good_matches)*0.5)]
            
        if len(good_matches) < 4:
            return None
            
        # Get points
        src_pts = np.float32([ self.prev_kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        # Find Homography
        M, mask_inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return None
            
        # Warp previous mask to new location
        new_mask = cv2.warpPerspective(self.prev_mask, M, (frame.shape[1], frame.shape[0]))
        
        # Filter new features that are inside the NEW mask
        new_kp_list = []
        new_des_list = []
        
        height, width = new_mask.shape
        for k, d in zip(kp, des):
            x, y = int(k.pt[0]), int(k.pt[1])
            if 0 <= x < width and 0 <= y < height:
                if new_mask[y, x] > 127:
                    new_kp_list.append(k)
                    new_des_list.append(d)
        
        if len(new_kp_list) > 4:
            self.prev_kp = new_kp_list
            self.prev_des = np.array(new_des_list)
            self.prev_mask = new_mask
            self.prev_frame = frame.copy()
        else:
            self.prev_mask = new_mask
            
        return new_mask

    def reset(self):
        self.prev_frame = None
        self.prev_mask = None
        self.prev_kp = None
        self.prev_des = None
        self.initialized = False


class ECCHomographyTracker(BaseTracker):
    """
    Homography Tracker with ECC Sub-pixel Refinement.
    """
    
    def __init__(self, feature_type="sift"):
        self.feature_type = feature_type
        if feature_type == "sift":
            self.detector = cv2.SIFT_create(nfeatures=1000)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Current frame state
        self.prev_frame = None
        self.prev_gray = None
        self.prev_mask = None
        self.prev_kp = None
        self.prev_des = None
        
        # Reference frame (for drift correction)
        self.reference_frame = None
        self.reference_gray = None
        self.reference_mask = None
        self.reference_kp = None
        self.reference_des = None
        self.accumulated_H = np.eye(3, dtype=np.float32)
        
        self.initialized = False
        self.frame_count = 0
        
        # ECC parameters
        self.ecc_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
    def init(self, frame: np.ndarray, mask: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        
        if kp is None or len(kp) < 4:
            print("ECCHomography: Not enough features in init.")
            return
        
        # Filter features inside mask
        good_kp, good_des = [], []
        h, w = mask.shape
        for k, d in zip(kp, des):
            x, y = int(k.pt[0]), int(k.pt[1])
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 127:
                good_kp.append(k)
                good_des.append(d)
        
        if len(good_kp) < 4:
            print("ECCHomography: Not enough features inside mask.")
            return
        
        # Store as both current and reference
        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()
        self.prev_mask = mask.copy()
        self.prev_kp = good_kp
        self.prev_des = np.array(good_des)
        
        self.reference_frame = frame.copy()
        self.reference_gray = gray.copy()
        self.reference_mask = mask.copy()
        self.reference_kp = good_kp.copy()
        self.reference_des = np.array(good_des)
        self.accumulated_H = np.eye(3, dtype=np.float32)
        
        self.initialized = True
        self.frame_count = 0
        print(f"ECCHomography initialized with {len(good_kp)} features.")
    
    def _match_and_find_homography(self, src_kp, src_des, dst_kp, dst_des):
        """Match features and compute homography."""
        if src_des is None or dst_des is None or len(src_des) < 4 or len(dst_des) < 4:
            return None
        
        if self.feature_type == "sift":
            matches = self.matcher.knnMatch(src_des, dst_des, k=2)
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        else:
            good_matches = self.matcher.match(src_des, dst_des)
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:len(good_matches)//2]
        
        if len(good_matches) < 4:
            return None
        
        src_pts = np.float32([src_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        
        if kp is None or len(kp) < 4:
            return None
        
        self.frame_count += 1
        
        # Frame-to-Frame homography
        H_f2f = self._match_and_find_homography(self.prev_kp, self.prev_des, kp, des)
        
        if H_f2f is None:
            return None
        
        # ECC Refinement for sub-pixel accuracy
        try:
            _, H_refined = cv2.findTransformECC(
                self.prev_gray, gray, H_f2f.astype(np.float32),
                cv2.MOTION_HOMOGRAPHY, self.ecc_criteria
            )
            H_f2f = H_refined
        except cv2.error:
            pass  # Use SIFT-only homography if ECC fails
        
        # Update accumulated homography
        self.accumulated_H = H_f2f @ self.accumulated_H
        
        # Every N frames, correct drift using reference frame
        if self.frame_count % 30 == 0 and self.reference_des is not None:
            H_ref = self._match_and_find_homography(
                self.reference_kp, self.reference_des, kp, des
            )
            if H_ref is not None:
                # Blend accumulated with reference-based (drift correction)
                alpha = 0.3
                self.accumulated_H = alpha * H_ref + (1 - alpha) * self.accumulated_H
        
        # Warp reference mask to current frame
        h, w = frame.shape[:2]
        new_mask = cv2.warpPerspective(self.reference_mask, self.accumulated_H, (w, h))
        
        # Update frame-to-frame state
        new_kp, new_des = [], []
        for k, d in zip(kp, des):
            x, y = int(k.pt[0]), int(k.pt[1])
            if 0 <= x < w and 0 <= y < h and new_mask[y, x] > 127:
                new_kp.append(k)
                new_des.append(d)
        
        if len(new_kp) >= 4:
            self.prev_kp = new_kp
            self.prev_des = np.array(new_des)
        
        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()
        self.prev_mask = new_mask
        
        return new_mask
    
    def reset(self):
        self.prev_frame = None
        self.prev_gray = None
        self.prev_mask = None
        self.prev_kp = None
        self.prev_des = None
        self.reference_frame = None
        self.reference_gray = None
        self.reference_mask = None
        self.reference_kp = None
        self.reference_des = None
        self.accumulated_H = np.eye(3, dtype=np.float32)
        self.initialized = False
        self.frame_count = 0


class SAM2MemoryTracker(BaseTracker):
    """
    SAM2 with Memory Bank and Distractor-Aware Memory (SAM2++).
    """
    
    def __init__(self, model_path: str = "sam2_b.pt", memory_size: int = 5):
        from ultralytics import SAM
        from collections import deque
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAM(model_path).to(self.device)
        self.memory_size = memory_size
        
        # Memory bank (stores frame, mask, embedding tuples)
        self.memory_bank = deque(maxlen=memory_size)
        
        # Reference (first high-quality detection)
        self.reference_frame = None
        self.reference_mask = None
        self.reference_embedding = None
        self.reference_bbox = None
        
        # Current state
        self.prev_frame = None
        self.prev_mask = None
        self.prev_bbox = None
        
        # Motion history for linear prediction
        self.motion_history = deque(maxlen=5)
        
        self.initialized = False
        self.occlusion_count = 0
        print(f"SAM2 Memory Tracker initialized on {self.device}")
    
    def _compute_embedding(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute color histogram embedding for masked region."""
        mask_u8 = (mask > 127).astype(np.uint8)
        if mask_u8.sum() == 0:
            return np.zeros(512)
        
        masked_region = cv2.bitwise_and(frame, frame, mask=mask_u8)
        hist = cv2.calcHist([masked_region], [0, 1, 2], mask_u8, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        return hist / (hist.sum() + 1e-6)
    
    def _is_distractor(self, embedding: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if current embedding differs significantly from reference."""
        if self.reference_embedding is None:
            return False
        
        # Histogram intersection
        similarity = np.minimum(embedding, self.reference_embedding).sum()
        return similarity < threshold
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Optional[list]:
        """Extract bounding box from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(mask.shape[1] - x, w + 2 * pad)
        h = min(mask.shape[0] - y, h + 2 * pad)
        return [x, y, x + w, y + h]
    
    def _linear_predict_bbox(self) -> Optional[list]:
        """Predict bbox based on motion history."""
        if len(self.motion_history) < 2:
            return self.prev_bbox
        
        # Compute average motion
        bboxes = list(self.motion_history)
        motion = np.array(bboxes[-1]) - np.array(bboxes[-2])
        
        if self.prev_bbox is None:
            return None
        
        predicted = np.array(self.prev_bbox) + motion
        return predicted.tolist()
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        self.reference_frame = frame.copy()
        self.reference_mask = mask.copy()
        self.reference_embedding = self._compute_embedding(frame, mask)
        self.reference_bbox = self._get_bbox_from_mask(mask)
        
        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        self.prev_bbox = self.reference_bbox
        
        if self.prev_bbox:
            self.motion_history.append(self.prev_bbox)
        
        self.memory_bank.clear()
        self.memory_bank.append((frame.copy(), mask.copy(), self.reference_embedding.copy()))
        
        self.initialized = True
        self.occlusion_count = 0
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_mask is None:
            return None
        
        # Get bbox for prompting
        bbox = self._get_bbox_from_mask(self.prev_mask)
        if bbox is None:
            bbox = self._linear_predict_bbox()
        
        if bbox is None:
            return None
        
        # Use SAM2 with bbox prompt
        results = self.model.predict(frame, bboxes=[bbox], device=self.device, verbose=False)
        
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
            tracked_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Compute embedding
            embedding = self._compute_embedding(frame, tracked_mask)
            
            # Check for distractor
            if self._is_distractor(embedding):
                self.occlusion_count += 1
                if self.occlusion_count > 3:
                    # Use linear prediction fallback
                    predicted_bbox = self._linear_predict_bbox()
                    if predicted_bbox:
                        # Try with predicted bbox
                        results = self.model.predict(frame, bboxes=[predicted_bbox], 
                                                    device=self.device, verbose=False)
                        if results and results[0].masks is not None:
                            mask = results[0].masks.data[0].cpu().numpy()
                            if mask.shape != frame.shape[:2]:
                                mask = cv2.resize(mask.astype(np.float32), 
                                                 (frame.shape[1], frame.shape[0]))
                            tracked_mask = (mask > 0.5).astype(np.uint8) * 255
                            embedding = self._compute_embedding(frame, tracked_mask)
            else:
                self.occlusion_count = 0
            
            # Update memory bank
            self.memory_bank.append((frame.copy(), tracked_mask.copy(), embedding.copy()))
            
            # Update motion history
            new_bbox = self._get_bbox_from_mask(tracked_mask)
            if new_bbox:
                self.motion_history.append(new_bbox)
            
            self.prev_frame = frame.copy()
            self.prev_mask = tracked_mask
            self.prev_bbox = new_bbox
            
            return tracked_mask
        
        return None
    
    def reset(self):
        self.memory_bank.clear()
        self.motion_history.clear()
        self.reference_frame = None
        self.reference_mask = None
        self.reference_embedding = None
        self.reference_bbox = None
        self.prev_frame = None
        self.prev_mask = None
        self.prev_bbox = None
        self.initialized = False
        self.occlusion_count = 0


class HybridFlowTracker(BaseTracker):
    """
    Hybrid Optical Flow + Homography Tracker with Forward-Backward Error.
    """
    
    def __init__(self):
        self.prev_gray = None
        self.prev_mask = None
        self.prev_points = None
        
        self.reference_gray = None
        self.reference_mask = None
        self.reference_points = None
        self.accumulated_H = np.eye(3, dtype=np.float32)
        
        self.initialized = False
        self.frame_count = 0
        
        # LK parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # GFTT parameters
        self.gftt_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10
        )
    
    def _get_gftt_points(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Get Good Features To Track inside mask."""
        mask_u8 = (mask > 127).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(gray, mask=mask_u8, **self.gftt_params)
        return points
    
    def init(self, frame: np.ndarray, mask: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = self._get_gftt_points(gray, mask)
        
        if points is None or len(points) < 4:
            print("HybridFlow: Not enough features in mask.")
            return
        
        self.prev_gray = gray.copy()
        self.prev_mask = mask.copy()
        self.prev_points = points
        
        self.reference_gray = gray.copy()
        self.reference_mask = mask.copy()
        self.reference_points = points.copy()
        self.accumulated_H = np.eye(3, dtype=np.float32)
        
        self.initialized = True
        self.frame_count = 0
        print(f"HybridFlow initialized with {len(points)} features.")
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.prev_points is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        self.frame_count += 1
        
        # Forward tracking
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if p1 is None:
            return None
        
        # Backward tracking (for FB error)
        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, p1, None, **self.lk_params
        )
        
        if p0_back is None:
            return None
        
        # Forward-backward error
        fb_error = np.linalg.norm(self.prev_points - p0_back, axis=2).flatten()
        
        # Filter by status and FB error
        good_mask = (st1.flatten() == 1) & (st2.flatten() == 1) & (fb_error < 1.0)
        
        if good_mask.sum() < 4:
            return None
        
        good_old = self.prev_points[good_mask]
        good_new = p1[good_mask]
        
        # Compute homography (planar constraint)
        H_f2f, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        
        if H_f2f is None:
            return None
        
        # Update accumulated homography
        self.accumulated_H = H_f2f @ self.accumulated_H
        
        # Drift correction every N frames
        if self.frame_count % 30 == 0 and self.reference_points is not None:
            # Try to match current frame with reference
            ref_points, st, _ = cv2.calcOpticalFlowPyrLK(
                self.reference_gray, gray, self.reference_points, None, **self.lk_params
            )
            
            if ref_points is not None and st.sum() >= 4:
                valid = st.flatten() == 1
                if valid.sum() >= 4:
                    H_ref, _ = cv2.findHomography(
                        self.reference_points[valid], ref_points[valid], cv2.RANSAC, 5.0
                    )
                    if H_ref is not None:
                        # Blend to correct drift
                        alpha = 0.3
                        self.accumulated_H = alpha * H_ref + (1 - alpha) * self.accumulated_H
        
        # Warp reference mask
        new_mask = cv2.warpPerspective(self.reference_mask, self.accumulated_H, (w, h))
        
        # Get new feature points for next frame
        new_points = self._get_gftt_points(gray, new_mask)
        
        if new_points is not None and len(new_points) >= 4:
            self.prev_points = new_points
        else:
            # Warp old points
            self.prev_points = cv2.perspectiveTransform(good_new.reshape(-1, 1, 2), 
                                                        np.eye(3, dtype=np.float32))
        
        self.prev_gray = gray.copy()
        self.prev_mask = new_mask
        
        return new_mask
    
    def reset(self):
        self.prev_gray = None
        self.prev_mask = None
        self.prev_points = None
        self.reference_gray = None
        self.reference_mask = None
        self.reference_points = None
        self.accumulated_H = np.eye(3, dtype=np.float32)
        self.initialized = False
        self.frame_count = 0


class ByteTracker(BaseTracker):
    """
    ByteTrack using Ultralytics (YOLO/RT-DETR).
    """
    
    def __init__(self, model_path="yolov8m.pt"):
        from ultralytics import YOLO, RTDETR
        if "rtdetr" in model_path.lower():
             self.model = RTDETR(model_path)
        else:
             self.model = YOLO(model_path)
             
        self.target_id = None
        self.initialized = False
        self.roi_mask = None
        
    def init(self, frame: np.ndarray, mask: np.ndarray):
        # Run tracking on first frame to get IDs
        results = self.model.track(frame, persist=True, verbose=False)
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            print("ByteTracker: No objects detected in init frame.")
            return

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        
        # simple IoU to match mask
        ys, xs = np.where(mask > 127)
        if len(xs) == 0: return
        
        mask_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        
        best_iou = 0
        best_id = -1
        
        for box, track_id in zip(boxes, ids):
            # Compute IoU
            xA = max(mask_box[0], box[0])
            yA = max(mask_box[1], box[1])
            xB = min(mask_box[2], box[2])
            yB = min(mask_box[3], box[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxArea = (box[2] - box[0]) * (box[3] - box[1])
            maskArea = (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])
            
            iou = interArea / float(boxArea + maskArea - interArea)
            
            if iou > best_iou:
                best_iou = iou
                best_id = track_id
                
        if best_iou > 0.1:
            self.target_id = best_id
            self.initialized = True
            print(f"ByteTracker initialized. Tracking ID: {self.target_id}")
            self.roi_mask = mask.copy()
        else:
            print("ByteTracker: Could not match mask to any track ID.")

    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.target_id is None:
            return None
            
        results = self.model.track(frame, persist=True, verbose=False)
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            return None
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        
        target_box = None
        for box, track_id in zip(boxes, ids):
            if track_id == self.target_id:
                target_box = box
                break
                
        if target_box is None:
            return None
            
        # Return rectangular mask for the box
        x1, y1, x2, y2 = target_box.astype(int)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # Clip
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask

    def reset(self):
        self.target_id = None
        self.initialized = False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_tracker(config) -> BaseTracker:
    """
    Factory function to create tracker based on config.
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        BaseTracker instance
    """
    tracker_type = config.tracker
    
    # Trackers
    if tracker_type == "sam2":
        return SAM2Tracker(model_path=config.sam2_model_path)
    elif tracker_type == "optical-flow":
        return AdaptiveOpticalFlowTracker(pyramid_levels=3)
    elif tracker_type == "bytetrack":
        # Prefer segmentation model for masks, otherwise fallback to detection
        model_path = "models/yolov8m_seg_best.pt"
        if not os.path.exists(model_path):
            model_path = config.det_path if (config.det_path and config.det_path.endswith('.pt')) else "yolov8m-seg.pt"
        return ByteTracker(model_path=model_path)
    elif tracker_type == "feature-homography":
        return FeatureHomographyTracker(feature_type="sift")
    elif tracker_type == "kalman":
        return PlanarKalmanTracker()
    
    # Advanced trackers
    elif tracker_type == "ecc-homography":
        return ECCHomographyTracker(feature_type="sift")
    elif tracker_type == "planar-kalman":
        return PlanarKalmanTracker()
    elif tracker_type == "sam2-memory":
        return SAM2MemoryTracker(model_path=config.sam2_model_path)
    elif tracker_type == "hybrid-flow":
        return HybridFlowTracker()
    
    else:
        raise ValueError(f"Unknown tracker: {tracker_type}")


def create_fusion_tracker(config) -> FusionTracker:
    """
    Create a fusion tracker combining multiple tracking strategies.
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        FusionTracker instance
    """
    # Define trackers to fuse
    trackers = []
    
    # Always include SAM2 for accuracy
    trackers.append(("sam2", SAM2Tracker(model_path=config.sam2_model_path)))
    
    # Add optical flow for smoothness
    trackers.append(("optical-flow", AdaptiveOpticalFlowTracker(pyramid_levels=3)))
    
    # Add feature-based tracker for robustness
    trackers.append(("feature-homography", FeatureHomographyTracker(feature_type="sift")))
    
    # Weights: SAM2 highest, others moderate
    weights = [0.5, 0.3, 0.2]
    
    return FusionTracker(trackers, weights)


# Backward compatibility aliases
OpticalFlowTracker = AdaptiveOpticalFlowTracker
PlanarKalman = PlanarKalmanTracker
SAM2TrackerMemory = SAM2MemoryTracker