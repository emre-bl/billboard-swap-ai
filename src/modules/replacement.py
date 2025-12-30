"""
Enhanced Replacement Module for Billboard Content Replacement.

Features:
- Improved perspective transformation with better corner detection
- Multi-level blending for seamless integration
- Content-aware scaling and sharpening
- Advanced edge blending with distance transform
- Occlusion handling and content adjustment
"""
import numpy as np
import cv2
import warnings
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull


@dataclass
class ReplacementConfig:
    """Configuration for replacement engine."""
    
    # Smoothing settings
    smoothing_enabled: bool = True
    smoothing_method: str = "kalman"  # "kalman", "ema", or "iir"
    smoothing_alpha: float = 0.3
    smoothing_window: int = 5
    
    # Blending settings
    edge_blend_enabled: bool = True
    edge_blend_radius: int = 7
    edge_blend_method: str = "gaussian"  # "gaussian", "distance", or "multires"
    use_distance_transform: bool = True
    feather_width: int = 15
    
    # Content adjustment
    brightness_adjust: bool = True
    contrast_adjust: bool = True
    color_match_enabled: bool = False
    sharpen_replacement: bool = True
    
    # Perspective settings
    enforce_convexity: bool = True
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    corner_refinement_iterations: int = 3
    use_subpixel_corners: bool = True
    
    # Quality settings
    interpolation_method: str = "lanczos4"  # "linear", "cubic", "lanczos4"
    antialiasing: bool = True


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in clockwise order starting from top-left.
    
    Args:
        pts: Array of 4 points (4, 2)
        
    Returns:
        Ordered points (4, 2)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Calculate centroid
    centroid = pts.mean(axis=0)
    
    # Calculate angles from centroid
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    
    # Sort by angle
    sorted_idx = np.argsort(angles)
    
    # Get sorted points
    sorted_pts = pts[sorted_idx]
    
    # Ensure starting from top-left (minimum x+y)
    sums = sorted_pts.sum(axis=1)
    start_idx = np.argmin(sums)
    
    # Reorder to start from top-left
    rect = np.roll(sorted_pts, -start_idx, axis=0)
    
    # Ensure proper order (top-left, top-right, bottom-right, bottom-left)
    # Check if points need swapping
    if rect[1, 0] < rect[0, 0]:
        # Swap if not in correct horizontal order
        rect[[0, 1]] = rect[[1, 0]]
        rect[[2, 3]] = rect[[3, 2]]
    
    return rect


def get_subpixel_corners(mask: np.ndarray, initial_corners: np.ndarray) -> np.ndarray:
    """
    Refine corners to sub-pixel accuracy using cornerSubPix.
    
    Args:
        mask: Binary mask
        initial_corners: Initial corner estimates (4, 2)
        
    Returns:
        Refined corners with sub-pixel accuracy
    """
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Convert to grayscale
    gray = mask.copy()
    
    # Parameters for corner refinement
    winSize = (11, 11)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Reshape corners for cornerSubPix
    corners = initial_corners.reshape(-1, 1, 2).astype(np.float32)
    
    # Refine corners
    refined_corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
    
    return refined_corners.reshape(4, 2)


def adjust_content_lighting(source_img: np.ndarray, target_region: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
    """
    Adjust replacement image lighting to match target region.
    
    Args:
        source_img: Replacement image
        target_region: Target region from original frame
        mask: Binary mask of target region
        
    Returns:
        Adjusted replacement image
    """
    if source_img.shape != target_region.shape:
        target_resized = cv2.resize(target_region, (source_img.shape[1], source_img.shape[0]))
        mask_resized = cv2.resize(mask, (source_img.shape[1], source_img.shape[0]))
        mask_resized = (mask_resized > 127).astype(np.uint8)
    else:
        target_resized = target_region
        mask_resized = (mask > 127).astype(np.uint8)
    
    # Only adjust if we have valid pixels
    if mask_resized.sum() == 0:
        return source_img
    
    # Convert to LAB color space for better color matching
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB)
    
    # Calculate mean and std for each channel
    source_mean = []
    source_std = []
    target_mean = []
    target_std = []
    
    for i in range(3):
        source_channel = source_lab[:, :, i][mask_resized > 0]
        target_channel = target_lab[:, :, i][mask_resized > 0]
        
        if len(source_channel) > 0 and len(target_channel) > 0:
            source_mean.append(np.mean(source_channel))
            source_std.append(np.std(source_channel))
            target_mean.append(np.mean(target_channel))
            target_std.append(np.std(target_channel))
        else:
            source_mean.append(0)
            source_std.append(1)
            target_mean.append(0)
            target_std.append(1)
    
    # Adjust each channel
    adjusted_lab = np.zeros_like(source_lab)
    for i in range(3):
        channel = source_lab[:, :, i].astype(np.float32)
        
        if source_std[i] > 0:
            # Match mean and std
            channel = (channel - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
        
        # Clip to valid range
        if i == 0:  # L channel
            channel = np.clip(channel, 0, 255)
        else:  # A and B channels
            channel = np.clip(channel, 0, 255)
        
        adjusted_lab[:, :, i] = channel
    
    # Convert back to BGR
    adjusted_img = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return adjusted_img


def create_feathered_mask(mask: np.ndarray, feather_width: int = 15) -> np.ndarray:
    """
    Create feathered mask using distance transform.
    
    Args:
        mask: Binary mask
        feather_width: Feathering width in pixels
        
    Returns:
        Soft mask (0-1 float)
    """
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Calculate distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normalize and feather
    max_dist = np.max(dist_transform)
    if max_dist > 0:
        dist_normalized = dist_transform / max_dist
    else:
        dist_normalized = (mask > 0).astype(np.float32)
    
    # Create feathering based on distance from edge
    feather_mask = np.ones_like(dist_normalized, dtype=np.float32)
    
    # Find edges (where distance is small)
    edge_mask = dist_normalized < (feather_width / max_dist if max_dist > 0 else 0.1)
    
    # Feather edges
    if edge_mask.any():
        # For edge pixels, use distance-based alpha
        feather_mask[edge_mask] = dist_normalized[edge_mask] * (max_dist / feather_width)
        feather_mask[feather_mask > 1] = 1
    
    return feather_mask


def pyramid_blend(source: np.ndarray, target: np.ndarray, 
                  mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    Multi-resolution pyramid blending for seamless transitions.
    
    Args:
        source: Source image (replacement)
        target: Target image (original)
        mask: Blending mask (0-1 float)
        levels: Number of pyramid levels
        
    Returns:
        Blended image
    """
    # Generate Gaussian pyramid for source, target, and mask
    source_pyramid = [source.astype(np.float32)]
    target_pyramid = [target.astype(np.float32)]
    mask_pyramid = [mask.astype(np.float32)]
    
    for i in range(levels - 1):
        source_pyramid.append(cv2.pyrDown(source_pyramid[-1]))
        target_pyramid.append(cv2.pyrDown(target_pyramid[-1]))
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    
    # Generate Laplacian pyramids
    source_laplacian = [source_pyramid[-1]]
    target_laplacian = [target_pyramid[-1]]
    
    for i in range(levels - 1, 0, -1):
        source_expanded = cv2.pyrUp(source_pyramid[i])
        target_expanded = cv2.pyrUp(target_pyramid[i])
        
        # Ensure same size
        h, w = source_pyramid[i-1].shape[:2]
        if source_expanded.shape[:2] != (h, w):
            source_expanded = cv2.resize(source_expanded, (w, h))
        if target_expanded.shape[:2] != (h, w):
            target_expanded = cv2.resize(target_expanded, (w, h))
        
        source_laplacian.append(source_pyramid[i-1] - source_expanded)
        target_laplacian.append(target_pyramid[i-1] - target_expanded)
    
    source_laplacian = list(reversed(source_laplacian))
    target_laplacian = list(reversed(target_laplacian))
    
    # Blend each level
    blended_pyramid = []
    for i in range(levels):
        blended = (mask_pyramid[i][:, :, None] * source_laplacian[i] + 
                  (1 - mask_pyramid[i][:, :, None]) * target_laplacian[i])
        blended_pyramid.append(blended)
    
    # Reconstruct
    blended = blended_pyramid[0]
    for i in range(1, levels):
        blended = cv2.pyrUp(blended)
        
        # Ensure same size
        h, w = blended_pyramid[i].shape[:2]
        if blended.shape[:2] != (h, w):
            blended = cv2.resize(blended, (w, h))
        
        blended = blended + blended_pyramid[i]
    
    return np.clip(blended, 0, 255).astype(np.uint8)


class EnhancedKalmanCornerSmoother:
    """
    Enhanced Kalman filter for smoothing corner coordinates with velocity prediction.
    """
    
    def __init__(
        self, 
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        velocity_factor: float = 0.1
    ):
        # State: 16 values (x, y, vx, vy for each of 4 corners)
        self.state_dim = 16
        self.measurement_dim = 8
        self.measurement_noise = measurement_noise  # Store for adaptive noise
        
        self.kf = cv2.KalmanFilter(self.state_dim, self.measurement_dim)
        self.velocity_factor = velocity_factor
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        for i in range(8):
            self.kf.transitionMatrix[i, i + 8] = 1.0
        
        # Measurement matrix (we only measure positions)
        self.kf.measurementMatrix = np.zeros((self.measurement_dim, self.state_dim), dtype=np.float32)
        for i in range(8):
            self.kf.measurementMatrix[i, i] = 1.0
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * process_noise
        self.kf.processNoiseCov[8:, 8:] *= 0.1  # Less noise for velocities
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(self.measurement_dim, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 100
        
        self.initialized = False
        self.frame_count = 0
        self.last_measurement = None
    
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
            # Initialize with measurement
            self.kf.statePost[:8] = measurement.reshape(-1, 1)
            self.initialized = True
            self.last_measurement = measurement.copy()
            self.frame_count = 1
            return corners
        
        self.frame_count += 1
        
        # Check for large jumps (scene change or detection error)
        if self.last_measurement is not None:
            dist = np.linalg.norm(measurement - self.last_measurement)
            if dist > 200:  # Reset if corners moved too much
                self.kf.statePost[:8] = measurement.reshape(-1, 1)
                self.kf.statePost[8:] = 0
                self.last_measurement = measurement.copy()
                return corners
        
        # Predict
        prediction = self.kf.predict()
        
        # Adapt measurement noise based on consistency
        if self.frame_count > 10:
            innovation = measurement - prediction[:8].flatten()
            innovation_norm = np.linalg.norm(innovation)
            
            # Increase noise if innovation is large
            if innovation_norm > 50:
                scale = min(10.0, innovation_norm / 50.0)
                self.kf.measurementNoiseCov = np.eye(self.measurement_dim, dtype=np.float32) * (
                    self.measurement_noise * scale
                )
        
        # Correct with measurement
        corrected = self.kf.correct(measurement.reshape(-1, 1))
        
        # Extract smoothed corners and velocities
        smoothed_corners = corrected[:8].flatten().reshape(4, 2)
        
        # Apply velocity damping
        velocities = corrected[8:].flatten().reshape(4, 2)
        smoothed_corners += velocities * self.velocity_factor
        
        self.last_measurement = measurement.copy()
        
        return smoothed_corners
    
    def reset(self):
        """Reset Kalman filter state."""
        self.initialized = False
        self.frame_count = 0
        self.last_measurement = None
        self.kf.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 100


class IIRCornersmoother:
    """
    Infinite Impulse Response (IIR) filter for corner smoothing.
    Multiple stages for better smoothing.
    """
    
    def __init__(self, alpha: float = 0.3, stages: int = 2):
        self.alpha = alpha
        self.stages = stages
        self.filters = []
        
        for _ in range(stages):
            self.filters.append({
                'prev_corners': None,
                'prev_velocity': np.zeros((4, 2), dtype=np.float32)
            })
    
    def update(self, corners: np.ndarray) -> np.ndarray:
        """
        Update IIR filter with new corners.
        
        Args:
            corners: Corner points (4, 2)
            
        Returns:
            Smoothed corners
        """
        current = corners.copy().astype(np.float32)
        
        for i, stage in enumerate(self.filters):
            if stage['prev_corners'] is None:
                stage['prev_corners'] = current.copy()
                continue
            
            # Calculate velocity
            velocity = current - stage['prev_corners']
            
            # Apply velocity smoothing
            smoothed_velocity = (self.alpha * velocity + 
                               (1 - self.alpha) * stage['prev_velocity'])
            
            # Update corners with smoothed velocity
            current = stage['prev_corners'] + smoothed_velocity
            
            # Store for next iteration
            stage['prev_corners'] = current.copy()
            stage['prev_velocity'] = smoothed_velocity.copy()
        
        return current
    
    def reset(self):
        """Reset all filter stages."""
        for stage in self.filters:
            stage['prev_corners'] = None
            stage['prev_velocity'] = np.zeros((4, 2), dtype=np.float32)


class EnhancedReplacementEngine:
    """
    Enhanced engine for replacing billboard content with perspective transformation.
    """
    
    def __init__(self, config: ReplacementConfig):
        self.config = config
        
        # Initialize smoother
        if config.smoothing_enabled:
            if config.smoothing_method == "kalman":
                self.smoother = EnhancedKalmanCornerSmoother()
            elif config.smoothing_method == "iir":
                self.smoother = IIRCornersmoother(alpha=config.smoothing_alpha, stages=3)
            else:
                self.smoother = EMACornerSmoother(alpha=config.smoothing_alpha)
        else:
            self.smoother = None
        
        # Cache for interpolation methods
        self.interpolation_flags = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        
        self.interpolation = self.interpolation_flags.get(
            config.interpolation_method, 
            cv2.INTER_LANCZOS4
        )
    
    def reset(self):
        """Reset engine state."""
        if self.smoother:
            self.smoother.reset()
    
    def extract_quadrilateral(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract quadrilateral from mask using improved algorithms.
        
        Args:
            mask: Binary mask
            
        Returns:
            4 corner points or None
        """
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Strategy 1: Try convex hull with adaptive epsilon
        hull = cv2.convexHull(cnt)
        
        # Use adaptive epsilon based on contour complexity
        epsilon_factor = 0.01
        best_approx = None
        best_score = float('inf')
        
        for factor in [0.005, 0.01, 0.02, 0.03, 0.04]:
            epsilon = factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                # Score based on area preservation
                area_ratio = cv2.contourArea(approx) / cv2.contourArea(cnt)
                score = abs(1.0 - area_ratio)
                
                if score < best_score:
                    best_score = score
                    best_approx = approx
        
        if best_approx is not None and len(best_approx) == 4:
            corners = best_approx.reshape(4, 2).astype(np.float32)
        else:
            # Strategy 2: Use minAreaRect with aspect ratio constraint
            rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(rect).astype(np.float32)
            
            # Check aspect ratio
            width = np.linalg.norm(corners[0] - corners[1])
            height = np.linalg.norm(corners[1] - corners[2])
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            if aspect_ratio > self.config.max_aspect_ratio or aspect_ratio < self.config.min_aspect_ratio:
                # Strategy 3: Use bounding quadrilateral from convex hull
                hull_points = hull.reshape(-1, 2)
                if len(hull_points) >= 4:
                    # Try to find 4 extreme points
                    rect_points = self._find_extreme_points(hull_points)
                    corners = rect_points.astype(np.float32)
        
        # Ensure convexity
        if self.config.enforce_convexity:
            corners = self._ensure_convexity(corners)
        
        # Order corners
        corners = order_points_clockwise(corners)
        
        # Sub-pixel refinement
        if self.config.use_subpixel_corners:
            for _ in range(self.config.corner_refinement_iterations):
                corners = get_subpixel_corners(mask, corners)
        
        return corners
    
    def _find_extreme_points(self, points: np.ndarray) -> np.ndarray:
        """
        Find 4 extreme points from a set of points.
        """
        # Find convex hull
        if len(points) < 4:
            hull_indices = ConvexHull(points).vertices
            hull_points = points[hull_indices]
        else:
            hull_points = points
        
        # Find extreme points in 8 directions (45-degree increments)
        angles = np.arange(0, 2 * np.pi, np.pi / 4)
        extreme_points = []
        
        center = np.mean(hull_points, axis=0)
        
        for angle in angles:
            # Project points onto direction vector
            direction = np.array([np.cos(angle), np.sin(angle)])
            projections = np.dot(hull_points - center, direction)
            
            # Find extreme point in this direction
            extreme_idx = np.argmax(projections)
            extreme_points.append(hull_points[extreme_idx])
        
        # Use k-means to reduce to 4 points
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=0).fit(extreme_points)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        return centers
    
    def _ensure_convexity(self, corners: np.ndarray) -> np.ndarray:
        """
        Ensure quadrilateral is convex.
        """
        # Check convexity using cross product
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            # Cross product
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if cross < 0:
                # Not convex, reorder
                corners = corners[[0, 3, 2, 1]]
                break
        
        return corners
    
    def enhance_replacement_image(self, img: np.ndarray, target_brightness: float = 1.0) -> np.ndarray:
        """
        Enhance replacement image for better visual quality.
        
        Args:
            img: Replacement image
            target_brightness: Target brightness level
            
        Returns:
            Enhanced image
        """
        enhanced = img.copy()
        
        # Sharpen if enabled
        if self.config.sharpen_replacement:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]]) / 1.0
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Adjust brightness and contrast
        if self.config.brightness_adjust or self.config.contrast_adjust:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            if self.config.contrast_adjust:
                # CLAHE for local contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
            
            if self.config.brightness_adjust:
                # Adjust brightness
                l = np.clip(l.astype(np.float32) * target_brightness, 0, 255).astype(np.uint8)
            
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_perspective_transform(
        self,
        source_img: np.ndarray,
        dst_corners: np.ndarray,
        output_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transform with enhanced interpolation.
        
        Args:
            source_img: Source image to transform
            dst_corners: Destination corners (4, 2)
            output_size: Output image size (width, height)
            
        Returns:
            Tuple of (warped image, warped mask)
        """
        # Source corners (full image)
        h, w = source_img.shape[:2]
        src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Warp image with selected interpolation
        warped = cv2.warpPerspective(
            source_img,
            matrix,
            output_size,
            flags=self.interpolation,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            mask,
            matrix,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Apply antialiasing to mask if enabled
        if self.config.antialiasing:
            kernel = np.ones((3, 3), np.float32) / 9
            warped_mask = cv2.filter2D(warped_mask, -1, kernel)
        
        return warped, warped_mask
    
    def blend_images(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        corners: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Advanced blending of foreground onto background.
        
        Args:
            foreground: Foreground image (replacement)
            background: Background image (original frame)
            mask: Blending mask
            corners: Corner points for distance-based blending
            
        Returns:
            Blended image
        """
        if not self.config.edge_blend_enabled:
            # Simple overlay
            result = background.copy()
            mask_bool = mask > 127
            result[mask_bool] = foreground[mask_bool]
            return result
        
        # Create blending mask
        if self.config.use_distance_transform and corners is not None:
            # Create distance-based feathering
            blend_mask = create_feathered_mask(mask, self.config.feather_width)
        else:
            # Gaussian blur-based blending
            kernel_size = self.config.edge_blend_radius * 2 + 1
            blend_mask = cv2.GaussianBlur(
                mask.astype(np.float32),
                (kernel_size, kernel_size),
                0
            ) / 255.0
        
        # Ensure blend_mask is 3-channel for color images
        if len(foreground.shape) == 3:
            blend_mask = np.stack([blend_mask] * 3, axis=-1)
        
        # Apply multi-resolution blending if selected
        if self.config.edge_blend_method == "multires" and len(foreground.shape) == 3:
            result = pyramid_blend(
                foreground.astype(np.float32),
                background.astype(np.float32),
                blend_mask[:, :, 0] if blend_mask.ndim == 3 else blend_mask,
                levels=5
            )
        else:
            # Alpha blending
            result = (foreground.astype(np.float32) * blend_mask + 
                     background.astype(np.float32) * (1 - blend_mask))
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def apply(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        replacement_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply billboard replacement with enhanced quality.
        
        Args:
            frame: Original frame
            mask: Binary mask of billboard
            replacement_img: Replacement image
            
        Returns:
            Tuple of (result frame, corner points)
        """
        # Extract quadrilateral from mask
        corners = self.extract_quadrilateral(mask)
        
        if corners is None:
            return frame, None
        
        # Apply temporal smoothing
        if self.smoother:
            corners = self.smoother.update(corners)
        
        # Enhance replacement image
        enhanced_replacement = self.enhance_replacement_image(replacement_img)
        
        # Apply perspective transform
        warped_replacement, warped_mask = self.apply_perspective_transform(
            enhanced_replacement,
            corners,
            (frame.shape[1], frame.shape[0])
        )
        
        # Adjust lighting if enabled
        if self.config.color_match_enabled:
            # Extract target region for color matching
            target_region = frame.copy()
            target_mask = (warped_mask > 127).astype(np.uint8)
            target_region[target_mask == 0] = 0
            
            warped_replacement = adjust_content_lighting(
                warped_replacement,
                target_region,
                warped_mask
            )
        
        # Blend images
        result = self.blend_images(
            warped_replacement,
            frame,
            warped_mask,
            corners
        )
        
        return result, corners


# Backward compatibility classes
class EMACornerSmoother:
    """Exponential Moving Average smoother for corners."""
    
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


class CornerSmoother(EMACornerSmoother):
    """Backward compatible alias for EMACornerSmoother."""
    pass


def apply_warp(
    frame: np.ndarray,
    mask: np.ndarray,
    replacement_img: np.ndarray,
    smoother: Optional[EMACornerSmoother] = None,
    config: Optional[ReplacementConfig] = None
) -> np.ndarray:
    """
    Enhanced backward compatible function for warp application.
    
    Args:
        frame: Original frame
        mask: Binary mask
        replacement_img: Replacement image
        smoother: Corner smoother (optional)
        config: Replacement configuration (optional)
        
    Returns:
        Result frame with replacement
    """
    # Use default config if not provided
    if config is None:
        config = ReplacementConfig()
    
    # Create engine
    engine = EnhancedReplacementEngine(config)
    
    # Set smoother if provided
    if smoother:
        if isinstance(smoother, EMACornerSmoother):
            engine.smoother = smoother
        else:
            warnings.warn(f"Unsupported smoother type: {type(smoother)}. Using default.")
    
    # Apply replacement
    result, _ = engine.apply(frame, mask, replacement_img)
    
    return result


# Backward compatibility aliases
ReplacementEngine = EnhancedReplacementEngine
KalmanCornerSmoother = EnhancedKalmanCornerSmoother
order_points = order_points_clockwise
