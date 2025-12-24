from ultralytics import SAM
import numpy as np
import cv2

class Segmenter:
    def __init__(self, model_path="sam2_b.pt"):
        # Supports SAM2 via Ultralytics
        self.model = SAM(model_path)
    
    def segment_bbox(self, frame, bboxes):
        """
        Refine segmentation using bounding boxes as prompts.
        bboxes: list of [x1, y1, x2, y2]
        """
        if len(bboxes) == 0:
            return []
        
        # SAM predict expects bboxes. 
        # API: results = model.predict(source, bboxes=...)
        results = self.model.predict(frame, bboxes=bboxes, verbose=False)
        masks = []
        if results and results[0].masks:
            # masks.data is usually (N, H, W)
            masks = results[0].masks.data.cpu().numpy()
        
        return masks

class GroundedSAM:
    def __init__(self, sam_model_path="sam2_b.pt", grounding_model_path="yolov8s-world.pt"):
        from ultralytics import YOLO, SAM
        self.grounding_model = YOLO(grounding_model_path)
        self.sam_model = SAM(sam_model_path)
    
    def detect_and_segment(self, frame, text_prompt="billboard"):
        """
        Detects objects matching the text_prompt and segments them.
        """
        # 1. Detect using YOLO-World (Zero-Shot)
        # Set classes based on text prompt? 
        # YOLO-World allows setting custom classes: model.set_classes(["billboard"])
        self.grounding_model.set_classes([text_prompt])
        
        results = self.grounding_model.predict(frame, verbose=False)
        bboxes = []
        if results and results[0].boxes:
            # Convert to list to avoid device mismatch/tensor issues
            bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        
        if len(bboxes) == 0:
            return [], [] # masks, bboxes
        
        # 2. Segment using SAM (Prompted by bboxes)
        sam_results = self.sam_model.predict(frame, bboxes=bboxes, verbose=False)
        masks = []
        if sam_results and sam_results[0].masks:
            masks = sam_results[0].masks.data.cpu().numpy()
            
        return masks, bboxes
