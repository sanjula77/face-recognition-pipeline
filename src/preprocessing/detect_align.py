"""
Face detection and alignment utilities using InsightFace or MTCNN
"""
import os

# Fix OpenMP library conflict on Windows
# This is needed when multiple libraries (NumPy, SciPy, InsightFace) use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path

# Check for InsightFace availability
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

# Check for MTCNN availability
try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

def init_insightface_detector(ctx_id=-1):
    """
    Initialize InsightFace detector (RetinaFace).
    
    Args:
        ctx_id: Device ID (0 for GPU, -1 for CPU)
    
    Returns:
        Initialized FaceAnalysis app
    """
    if not HAS_INSIGHTFACE:
        raise ImportError("InsightFace is not installed. Install with: pip install insightface")
    
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

def init_mtcnn(device='cpu'):
    """
    Initialize MTCNN detector.
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        Initialized MTCNN detector
    """
    if not HAS_MTCNN:
        raise ImportError("MTCNN is not installed. Install with: pip install mtcnn")
    
    return MTCNN()

def detect_faces_insightface(app, img, min_face_width=50):
    """
    Detect faces using InsightFace.
    
    Args:
        app: Initialized FaceAnalysis app
        img: Input image (BGR format)
        min_face_width: Minimum face width in pixels
    
    Returns:
        List of face dictionaries with 'bbox', 'landmarks', and 'score'
    """
    if not HAS_INSIGHTFACE:
        raise ImportError("InsightFace is not installed")
    
    # Convert BGR to RGB for InsightFace
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        width = bbox[2] - bbox[0]
        
        if width < min_face_width:
            continue
        
        # Convert landmarks from InsightFace format to 5-point format
        # InsightFace provides 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        landmarks = face.kps.astype(int)  # Shape: (5, 2)
        
        results.append({
            'bbox': bbox.tolist(),
            'landmarks': landmarks.tolist(),
            'score': float(face.det_score)
        })
    
    return results

def detect_faces_mtcnn(detector, img, min_face_width=50):
    """
    Detect faces using MTCNN.
    
    Args:
        detector: Initialized MTCNN detector
        img: Input image (BGR format)
        min_face_width: Minimum face width in pixels
    
    Returns:
        List of face dictionaries with 'bbox', 'landmarks', and 'score'
    """
    if not HAS_MTCNN:
        raise ImportError("MTCNN is not installed")
    
    # Convert BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)
    
    results = []
    for detection in detections:
        bbox = detection['box']  # [x, y, width, height]
        width = bbox[2]
        
        if width < min_face_width:
            continue
        
        # Convert to [x1, y1, x2, y2] format
        x, y, w, h = bbox
        bbox_xyxy = [x, y, x + w, y + h]
        
        # Extract landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        keypoints = detection['keypoints']
        landmarks = np.array([
            [keypoints['left_eye'][0], keypoints['left_eye'][1]],
            [keypoints['right_eye'][0], keypoints['right_eye'][1]],
            [keypoints['nose'][0], keypoints['nose'][1]],
            [keypoints['mouth_left'][0], keypoints['mouth_left'][1]],
            [keypoints['mouth_right'][0], keypoints['mouth_right'][1]]
        ])
        
        results.append({
            'bbox': bbox_xyxy,
            'landmarks': landmarks.tolist(),
            'score': float(detection['confidence'])
        })
    
    return results

def align_face(img, landmarks, output_size=(112, 112)):
    """
    Align face using 5-point landmarks.
    
    Args:
        img: Input image (BGR format)
        landmarks: 5-point landmarks [[x1,y1], [x2,y2], ...] for:
                   left_eye, right_eye, nose, left_mouth, right_mouth
        output_size: Output image size (width, height)
    
    Returns:
        Aligned face image
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Standard face alignment reference points (for 112x112 output)
    # These are the expected positions for the 5 landmarks in the aligned image
    ref_landmarks = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.5014],  # Right eye
        [48.0252, 71.7366],  # Nose
        [33.5493, 92.3655],  # Left mouth
        [62.7299, 92.2041]   # Right mouth
    ], dtype=np.float32)
    
    # Scale reference landmarks to output size
    scale_x = output_size[0] / 112.0
    scale_y = output_size[1] / 112.0
    ref_landmarks[:, 0] *= scale_x
    ref_landmarks[:, 1] *= scale_y
    
    # Calculate affine transformation matrix
    transform_matrix = cv2.getAffineTransform(
        landmarks[:3].astype(np.float32),  # Use first 3 points (eyes and nose)
        ref_landmarks[:3].astype(np.float32)
    )
    
    # Apply affine transformation
    aligned = cv2.warpAffine(img, transform_matrix, output_size, flags=cv2.INTER_LINEAR)
    
    return aligned

def apply_clahe(img, clip=2.0, tile=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img: Input image (BGR format)
        clip: Clip limit for CLAHE
        tile: Tile grid size for CLAHE
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def compute_sharpness(img):
    """
    Compute image sharpness using Laplacian variance.
    
    Args:
        img: Input image
    
    Returns:
        Sharpness score
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return float(sharpness)

def compute_brightness(img):
    """
    Compute average image brightness.
    
    Args:
        img: Input image
    
    Returns:
        Average brightness value
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return float(brightness)

def normalize_for_arcface(img):
    """
    Normalize image for ArcFace model input.
    ArcFace expects images in range [0, 1] with specific normalization.
    
    Args:
        img: Input image (BGR, uint8)
    
    Returns:
        Normalized image (float32, range [0, 1])
    """
    # Convert to float and normalize to [0, 1]
    normalized = img.astype(np.float32) / 255.0
    
    # ArcFace typically expects RGB format, so convert BGR to RGB
    normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    return normalized_rgb

def save_normalized_face(normalized_face, filepath):
    """
    Save normalized face as .npy file.
    
    Args:
        normalized_face: Normalized face array
        filepath: Output file path
    """
    np.save(filepath, normalized_face)

