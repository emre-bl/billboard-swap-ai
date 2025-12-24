from ultralytics import YOLO
import numpy as np
import torch
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_model(num_classes):
    # Load pre-trained Mask R-CNN
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

class BillboardDetector:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, frame, conf=0.4):
        """
        Detects billboards in the frame.
        Returns a list of bounding boxes [x1, y1, x2, y2].
        """
        results = self.model.predict(frame, conf=conf, verbose=False)
        bboxes = []
        if results and results[0].boxes:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
        return bboxes

    def segment(self, frame, conf=0.4):
        """
        Returns masks from YOLO-seg if available.
        """
        results = self.model.predict(frame, conf=conf, verbose=False)
        masks = []
        if results and results[0].masks:
            masks = results[0].masks.data.cpu().numpy()
        return masks

class MaskRCNNDetector:
    def __init__(self, model_path, num_classes=2): # 1 class + background
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_maskrcnn_model(num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def segment(self, frame, conf=0.5):
        """
        Returns masks using Mask R-CNN.
        """
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
            
        masks = []
        # Filter by confidence
        scores = predictions['scores'].cpu().numpy()
        high_conf_indices = np.where(scores > conf)[0]
        
        if len(high_conf_indices) > 0:
            # Mask R-CNN returns soft masks [0, 1], need to threshold
            raw_masks = predictions['masks'][high_conf_indices].cpu().numpy()
            # Remove channel dim: (N, 1, H, W) -> (N, H, W)
            raw_masks = raw_masks.squeeze(1)
            # Threshold to binary
            masks = (raw_masks > 0.5).astype(np.uint8)
            
        return masks

class HybridDetector:
    def __init__(self, yolo_path="yolov8n-seg.pt", sam_path="sam2_b.pt"):
        from ultralytics import YOLO, SAM
        self.detector = YOLO(yolo_path)
        self.segmenter = SAM(sam_path)
        
    def detect_and_segment(self, frame, conf=0.4):
        # 1. Detect Box with YOLO
        results = self.detector.predict(frame, conf=conf, verbose=False)
        if not results:
             return []
        
        # Check if boxes exist
        if not hasattr(results[0], 'boxes') or not results[0].boxes:
             return []
        
        # 2. Refine with SAM (using boxes prompt)
        bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        if not bboxes: return []
        
        sam_results = self.segmenter.predict(frame, bboxes=bboxes, verbose=False)
        masks = []
        if sam_results and sam_results[0].masks:
             masks = sam_results[0].masks.data.cpu().numpy()
             
        return masks
