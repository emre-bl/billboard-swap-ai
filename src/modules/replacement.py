import cv2
import numpy as np

def order_points(pts):
    """Orders 4 points as TL, TR, BR, BL for perspective warping."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

class CornerSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_corners = None
        
    def update(self, corners):
        """
        Smooths corner points using Exponential Moving Average (EMA).
        corners: (4, 2) array
        """
        if self.prev_corners is None:
            self.prev_corners = corners
            return corners
        
        # Calculate distance between new and old corners to detect "jumps" (e.g. tracking failure)
        # If jump is too large, reset or accept new
        dist = np.linalg.norm(self.prev_corners - corners)
        if dist > 50.0: # Reset if moved too much (scene change or detection error)
             self.prev_corners = corners
             return corners
             
        smoothed = self.prev_corners * self.alpha + corners * (1 - self.alpha)
        self.prev_corners = smoothed
        return smoothed

def apply_warp(frame, mask, replacement_img, smoother=None):
    """Warps replacement_img into the area defined by the mask.
       smoother: Optional CornerSmoother instance
    """
    # Find contours and approximate to 4 points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return frame
    
    # Identify the largest contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify contour to 4 points (the corners of the billboard)
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if len(approx) != 4:
        # Fallback: Use bounding box if approx fails
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect)
    
    dst_pts = order_points(approx.reshape(4, 2))
    
    # Apply Smoothing
    if smoother:
        dst_pts = smoother.update(dst_pts)
        
    h, w = replacement_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Calculate Homography Matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(replacement_img, matrix, (frame.shape[1], frame.shape[0]))
    
    # Create mask for the warped region to blend
    mask_indices = np.where(warped > 0)
    frame[mask_indices] = warped[mask_indices]
    return frame
