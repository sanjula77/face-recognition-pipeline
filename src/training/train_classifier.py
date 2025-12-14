"""
Training utilities for face recognition classifiers
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, List

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.utils import load_embeddings_database

def load_embeddings_from_dir(emb_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load embeddings from directory and return X, y, labels for training.
    
    Args:
        emb_dir: Directory containing embedding files
    
    Returns:
        Tuple of (X, y, labels) where:
        - X: embedding matrix (n_samples, n_features)
        - y: label array (n_samples,) with person names as strings
        - labels: list of unique person names
    """
    return load_embeddings_database(emb_dir)

class ArcFaceWrapper:
    """
    Wrapper for ArcFace model to extract embeddings from images.
    """
    
    def __init__(self, model_name: str = 'buffalo_l', ctx_id: int = -1):
        """
        Initialize ArcFace model.
        
        Args:
            model_name: Model name (buffalo_l, buffalo_s, etc.)
            ctx_id: Device ID (0 for GPU, -1 for CPU)
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.model_name = model_name
            self.ctx_id = ctx_id
        except ImportError:
            raise ImportError("InsightFace is not installed. Install with: pip install insightface")
    
    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an image.
        
        Args:
            img: Input image (BGR, uint8) or normalized face image
        
        Returns:
            Embedding vector (512-dimensional)
        """
        # If image is normalized [0,1], convert to uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
                # Convert RGB to BGR if needed
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Ensure correct shape
        if len(img.shape) == 3 and img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        
        # Use recognition model directly
        embedding = self.app.models['recognition'].get_feat(img)
        return embedding.flatten()

def precompute_augmented_embeddings(arcface: ArcFaceWrapper, images_dir: str, 
                                   n_augment: int = 1, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute embeddings from processed images with optional augmentation.
    
    Args:
        arcface: ArcFaceWrapper instance
        images_dir: Directory containing processed face images (.npy files)
        n_augment: Number of augmentations per image
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (X, y) where:
        - X: embedding matrix (n_samples, n_features)
        - y: label array (n_samples,) with person names as strings
    """
    images_path = Path(images_dir)
    
    # Find all .npy face files
    face_files = list(images_path.rglob("*.npy"))
    
    if not face_files:
        raise ValueError(f"No face files found in {images_dir}")
    
    print(f"📊 Processing {len(face_files)} face files")
    
    embeddings = []
    labels = []
    
    for face_file in face_files:
        try:
            # Load normalized face (RGB float32 [0,1])
            face_normalized = np.load(face_file)
            
            # Convert to BGR uint8 for ArcFace
            if face_normalized.dtype != np.uint8:
                face_uint8 = (face_normalized * 255.0).astype(np.uint8)
            else:
                face_uint8 = face_normalized
            
            # Convert RGB to BGR
            if len(face_uint8.shape) == 3:
                face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)
            else:
                face_bgr = face_uint8
            
            # Extract embedding
            embedding = arcface.get_embedding(face_bgr)
            embeddings.append(embedding)
            
            # Extract person name from path
            person_name = face_file.parent.name
            labels.append(person_name)
            
            # Apply augmentations if requested
            if n_augment > 1:
                for _ in range(n_augment - 1):
                    # Simple augmentation: slight rotation, brightness adjustment
                    augmented = apply_simple_augmentation(face_bgr)
                    aug_embedding = arcface.get_embedding(augmented)
                    embeddings.append(aug_embedding)
                    labels.append(person_name)
        
        except Exception as e:
            print(f"⚠️  Failed to process {face_file.name}: {e}")
            continue
    
    X = np.vstack(embeddings) if embeddings else np.array([])
    y = np.array(labels) if labels else np.array([])
    
    print(f"✅ Precomputed {X.shape[0]} embeddings from {len(set(labels))} people")
    
    return X, y

def apply_simple_augmentation(img: np.ndarray) -> np.ndarray:
    """
    Apply simple augmentation to image.
    
    Args:
        img: Input image (BGR, uint8)
    
    Returns:
        Augmented image
    """
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    img_aug = (img * brightness).clip(0, 255).astype(np.uint8)
    
    # Random slight rotation
    angle = np.random.uniform(-5, 5)
    h, w = img_aug.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_aug = cv2.warpAffine(img_aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return img_aug

