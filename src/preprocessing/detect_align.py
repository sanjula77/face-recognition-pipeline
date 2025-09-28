# src/preprocessing/detect_align.py
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import json
import math

# Try insightface (RetinaFace) first; fallback to facenet-pytorch MTCNN if needed
HAS_INSIGHTFACE = False
HAS_MTCNN = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception:
    HAS_INSIGHTFACE = False

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False

# canonical 5-point template for 112x112 (common ArcFace template)
_TEMPLATE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041]   # right mouth
], dtype=np.float32)

def init_insightface_detector(ctx_id: int = 0, det_size=(640,640)):
    """Initialize insightface FaceAnalysis (RetinaFace). ctx_id=0 tries GPU (if built), -1 CPU."""
    if not HAS_INSIGHTFACE:
        raise RuntimeError("insightface not installed")
    app = FaceAnalysis(allowed_modules=['detection','landmark'])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

def init_mtcnn(device='cpu', keep_all=True, thresholds=[0.6,0.7,0.7]):
    """Initialize MTCNN detector."""
    if not HAS_MTCNN:
        raise RuntimeError("facenet-pytorch not installed")
    mtcnn = MTCNN(keep_all=keep_all, device=device)
    return mtcnn

def compute_sharpness(img: np.ndarray) -> float:
    """Laplacian variance sharpness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_brightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def apply_clahe(img: np.ndarray, clip=2.0, tile=8) -> np.ndarray:
    """Apply CLAHE on L channel (LAB color)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute 2x3 similarity (affine) transform from src_pts (N,2) to dst_pts (N,2).
    Uses OpenCV estimateAffinePartial2D (similarity).
    """
    src = src_pts.astype(np.float32)
    dst = dst_pts.astype(np.float32)
    tform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if tform is None:
        raise RuntimeError("Could not compute transform")
    return tform

def align_face(img: np.ndarray, landmarks: np.ndarray, output_size=(112,112)) -> np.ndarray:
    """
    landmarks: 5 x 2 ordered as [left_eye, right_eye, nose, left_mouth, right_mouth]
    output_size: (w,h)
    """
    out_w, out_h = output_size
    # scale template to requested output size (template designed for 112x112)
    scale_x = out_w / 112.0
    scale_y = out_h / 112.0
    dst = _TEMPLATE_5PTS.copy()
    dst[:,0] *= scale_x
    dst[:,1] *= scale_y

    src = np.array(landmarks, dtype=np.float32)
    if src.shape[0] != 5:
        raise ValueError("align_face expects 5 landmarks")
    tform = estimate_similarity_transform(src, dst)
    aligned = cv2.warpAffine(img, tform, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return aligned

# Face detection wrapper
def detect_faces_insightface(app, img: np.ndarray, min_face_width_px:int=80):
    """
    Returns list of dicts:
      [{'bbox': [x1,y1,x2,y2], 'score': float, 'landmarks': np.array(5,2)} ...]
    """
    outs = []
    # app.get expects BGR (OpenCV) or RGB? insightface docs: expects BGR image (numpy)
    faces = app.get(img)
    for face in faces:
        # face.bbox is [x1,y1,x2,y2] floats
        bbox = np.array(face.bbox).astype(int).tolist()
        # try common landmark names
        # insightface Face object: has 'kps' or 'landmark_2d_5' depending on version
        lm = None
        if hasattr(face, 'kps') and face.kps is not None:
            lm = np.array(face.kps, dtype=np.float32)  # may be 5 pts
        elif hasattr(face, 'landmark_2d_5') and face.landmark_2d_5 is not None:
            lm = np.array(face.landmark_2d_5, dtype=np.float32)
        elif hasattr(face, 'landmark') and face.landmark is not None:
            lm = np.array(face.landmark, dtype=np.float32)
        else:
            # try to extract from face.__dict__
            d = face.__dict__
            for k in d:
                if 'landmark' in k or 'kps' in k:
                    try:
                        lm = np.array(d[k])
                        break
                    except Exception:
                        pass
        score = float(face.det_score) if hasattr(face, 'det_score') else float(face.score) if hasattr(face,'score') else 0.0
        if lm is None:
            lm = None
        # filter by size
        width = bbox[2] - bbox[0]
        if width < min_face_width_px:
            continue
        outs.append({'bbox': bbox, 'score': score, 'landmarks': lm})
    return outs

def detect_faces_mtcnn(mtcnn, img: np.ndarray, min_face_width_px:int=80):
    """
    MTCNN returns boxes and landmarks; convert to our format.
    """
    outs = []
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return []
    for box, p, lm in zip(boxes, probs, points):
        x1,y1,x2,y2 = map(int, box.tolist())
        width = x2 - x1
        if width < min_face_width_px:
            continue
        # points are 5x2 in order: left_eye, right_eye, nose, mouth_left, mouth_right
        outs.append({'bbox':[x1,y1,x2,y2], 'score': float(p), 'landmarks': np.array(lm, dtype=np.float32)})
    return outs

def normalize_for_arcface(image: np.ndarray) -> np.ndarray:
    """
    Normalize face image for ArcFace embedding extraction.
    
    Args:
        image: Input image (112x112 RGB, dtype uint8)
        
    Returns:
        Normalized image (3x112x112, dtype float32, range [-1, 1])
    """
    # Convert to float32
    img_float = image.astype(np.float32)
    
    # Scale from [0, 255] to [0, 1]
    img_normalized = img_float / 255.0
    
    # Normalize to [-1, 1] range
    img_arcface = (img_normalized - 0.5) / 0.5
    
    # Transpose from HWC to CHW format (PyTorch format)
    img_chw = np.transpose(img_arcface, (2, 0, 1))
    
    return img_chw

def save_normalized_face(normalized_array: np.ndarray, output_path: str):
    """
    Save normalized face array as .npy file.
    
    Args:
        normalized_array: Normalized face array (3x112x112, float32)
        output_path: Path to save the .npy file
    """
    np.save(output_path, normalized_array)

# Helper to save metadata next to processed image
def save_metadata_json(out_path: str, meta: dict):
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
