import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from ultralytics import YOLO

# --- 1. INITIALIZE THE 3 AI MODELS ---
model_yolo = YOLO('yolov8n-seg.pt')
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# --- 2. HELPER FUNCTIONS ---
def get_depth_blurred_bg(bg_image, midas_model, transforms, device):
    """Estimates depth map of background and applies depth-of-field blur."""
    img_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    input_batch = transforms(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=bg_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Variable depth blur
    blur_near = cv2.GaussianBlur(bg_image, (15, 15), 0)
    blur_far = cv2.GaussianBlur(bg_image, (51, 51), 0)

    depth_mask = (depth_map / 255.0)[:, :, np.newaxis]
    depth_aware_bg = (blur_near * depth_mask + blur_far * (1 - depth_mask)).astype(np.uint8)
    return depth_aware_bg

def process_frame(frame, depth_aware_bg):
    """Processes a single frame using ensembled masks and pre-computed background."""
    h, w, _ = frame.shape
    
    if depth_aware_bg.shape[:2] != (h, w):
        depth_aware_bg = cv2.resize(depth_aware_bg, (w, h))

    # YOLOv8 Mask
    yolo_mask = np.zeros((h, w), dtype=np.float32)
    results = model_yolo(frame, verbose=False)[0]
    if results.masks is not None:
        for box, mask in zip(results.boxes, results.masks):
            if int(box.cls[0]) == 0:  
                m = cv2.resize(mask.data[0].cpu().numpy(), (w, h))
                yolo_mask = np.maximum(yolo_mask, m)

    # MediaPipe Mask
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = mp_selfie.process(rgb_frame)
    mp_mask = mp_results.segmentation_mask if mp_results.segmentation_mask is not None else np.zeros((h, w), dtype=np.float32)

    # Ensemble & Blend
    combined_mask = (0.5 * yolo_mask) + (0.5 * mp_mask)
    binary_mask = np.where(combined_mask > 0.4, 1.0, 0.0).astype(np.float32)
    smooth_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)[:, :, np.newaxis]

    output_frame = (frame * smooth_mask + depth_aware_bg * (1 - smooth_mask)).astype(np.uint8)
    return output_frame

# --- 3. VIDEO PROCESSING PIPELINE ---
input_video_path = 'input_video.mp4'
bg_image_path = 'background.jpg'
output_video_path = 'output_virtual_bg.mp4'

bg_img = cv2.imread(bg_image_path) if os.path.exists(bg_image_path) else np.zeros((720, 1280, 3), dtype=np.uint8)
depth_aware_bg = get_depth_blurred_bg(bg_img, midas, midas_transforms, device)

if os.path.exists(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(process_frame(frame, depth_aware_bg))

    cap.release()
    out.release()
    print("Video processing complete.")
