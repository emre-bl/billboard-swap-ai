import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import tempfile
from PIL import Image

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

def apply_warp(frame, mask, replacement_img):
    """Warps replacement_img into the area defined by the mask."""
    # Find contours and approximate to 4 points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return frame
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify contour to 4 points (the corners of the billboard)
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if len(approx) != 4:
        # Fallback: Use bounding box if approx fails
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect)
    
    dst_pts = order_points(approx.reshape(4, 2))
    h, w = replacement_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Calculate Homography Matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(replacement_img, matrix, (frame.shape[1], frame.shape[0]))
    
    # Create mask for the warped region to blend
    mask_indices = np.where(warped > 0)
    frame[mask_indices] = warped[mask_indices]
    return frame

st.title("Billboard Changer - Emre Belikırık")

# Sidebar configs
st.sidebar.header("Models")
yolo_path = st.sidebar.text_input("YOLOv8 Weights", "runs/segment/billboard_segmentation/weights/best.pt")
sam_path = st.sidebar.selectbox("SAM2 Model", ["sam2.1_t.pt", "sam2.1_b.pt"])

# Uploaders
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])
replacement_file = st.sidebar.file_uploader("Upload Replacement Ad", type=["jpg", "png"])

if video_file and replacement_file:
    yolo_model = YOLO(yolo_path)
    sam_model = SAM(sam_path)
    
    # Load replacement image
    repl_img = Image.open(replacement_file)
    repl_img = cv2.cvtColor(np.array(repl_img), cv2.COLOR_RGB2BGR)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detection (YOLOv8-seg)
        results = yolo_model.predict(frame, conf=0.4)
        
        for r in results:
            if r.boxes:
                # 2. Refine with SAM2 (using YOLO bounding boxes)
                bboxes = r.boxes.xyxy.cpu().numpy()
                sam_results = sam_model.predict(frame, bboxes=bboxes)
                
                for s_res in sam_results:
                    if s_res.masks:
                        mask = s_res.masks.data[0].cpu().numpy().astype(np.uint8)
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # 3. Replace/Warp
                        frame = apply_warp(frame, mask, repl_img)
        
        st_frame.image(frame, channels="BGR")
    cap.release()