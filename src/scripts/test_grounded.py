import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.segmentation import GroundedSAM

def test_grounded_sam():
    print("Initializing GroundedSAM...")
    # Expecting models to be available or downloaded
    # User had sam2_b.pt in root. We might need to point to it if it's not in CWD.
    # The default paths in GroundedSAM are "sam2_b.pt" and "yolov8s-world.pt".
    
    # We should check if sam2_b.pt is in current directory or project root.
    # Current CWD when running run_command is project root.
    
    gs = GroundedSAM(sam_model_path="sam2_b.pt", grounding_model_path="yolov8s-world.pt")
    
    # Create dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a rectangle to substitute a billboard
    cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
    
    print("Running detection and segmentation...")
    masks, bboxes = gs.detect_and_segment(img, text_prompt="rectangle") # Prompt for "rectangle" since we drew one
    
    print(f"Detected {len(bboxes)} objects.")
    if len(masks) > 0:
        print(f"Mask shape: {masks[0].shape}")
    
    print("Test passed if no errors.")

if __name__ == "__main__":
    test_grounded_sam()
