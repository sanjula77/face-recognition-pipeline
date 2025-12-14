"""
Face Processor Module
Handles face detection, alignment, preprocessing, and embedding extraction using InsightFace.
Includes CLAHE, quality filtering, and normalization preprocessing steps.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
    print("⚠️  InsightFace not available. Install with: pip install insightface")

try:
    from src.preprocessing.face_quality import FaceQualityAssessor
    HAS_QUALITY_FILTER = True
except ImportError:
    HAS_QUALITY_FILTER = False


class FaceProcessor:
    """
    Processes images to detect faces, align them, preprocess, and extract embeddings.
    Uses RetinaFace for detection and ArcFace for embedding extraction.
    Includes preprocessing: CLAHE, quality filtering, and normalization.
    """
    
    def __init__(self, model_name: str = 'buffalo_l', ctx_id: int = -1,
                 enable_clahe: bool = True,
                 enable_quality_filter: bool = True,
                 min_quality: float = 0.5,
                 min_face_width: int = 50,
                 clahe_clip: float = 2.0,
                 clahe_tile: Tuple[int, int] = (8, 8)):
        """
        Initialize face processor.
        
        Args:
            model_name: InsightFace model name ('buffalo_l', 'buffalo_s', etc.)
            ctx_id: Device ID (-1 for CPU, 0+ for GPU)
            enable_clahe: Enable CLAHE preprocessing
            enable_quality_filter: Enable face quality filtering
            min_quality: Minimum quality score threshold (0-1)
            min_face_width: Minimum face width in pixels
            clahe_clip: CLAHE clip limit
            clahe_tile: CLAHE tile grid size
        """
        if not HAS_INSIGHTFACE:
            raise ImportError("InsightFace is required. Install with: pip install insightface")
        
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.app: Optional[FaceAnalysis] = None
        
        # Preprocessing settings
        self.enable_clahe = enable_clahe
        self.enable_quality_filter = enable_quality_filter
        self.min_quality = min_quality
        self.min_face_width = min_face_width
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        
        # Initialize quality assessor if available
        self.quality_assessor = None
        if self.enable_quality_filter and HAS_QUALITY_FILTER:
            try:
                self.quality_assessor = FaceQualityAssessor()
            except Exception:
                self.enable_quality_filter = False
                print("⚠️  Face quality filtering disabled (module not available)")
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize InsightFace FaceAnalysis app."""
        try:
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            print(f"✅ Face processor initialized: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize face processor: {e}")
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image with preprocessing.
        
        Args:
            image: Input image (BGR format, numpy array)
        
        Returns:
            List of face detection results, each containing:
            - bbox: Bounding box [x1, y1, x2, y2]
            - landmarks: Facial landmarks
            - score: Detection confidence score
        """
        if self.app is None:
            raise RuntimeError("Face processor not initialized")
        
        # Apply CLAHE preprocessing if enabled
        processed_image = image.copy()
        if self.enable_clahe:
            processed_image = self._apply_clahe(processed_image)
        
        # Convert BGR to RGB for InsightFace
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(image_rgb)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            width = bbox[2] - bbox[0]
            
            # Filter by minimum face width
            if width < self.min_face_width:
                continue
            
            # Extract landmarks
            landmarks = None
            if hasattr(face, 'kps'):
                landmarks = face.kps.astype(float).tolist()  # 5-point landmarks
            elif hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106.astype(float).tolist()
            
            results.append({
                'bbox': bbox.tolist(),
                'landmarks': landmarks,
                'score': float(face.det_score)
            })
        
        return results
    
    def extract_embedding(self, image: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image.
        If bbox is provided, extracts from that region; otherwise detects first face.
        
        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box [x1, y1, x2, y2]
        
        Returns:
            Face embedding vector (512-dimensional) or None if no face detected
        """
        if self.app is None:
            raise RuntimeError("Face processor not initialized")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(image_rgb)
        
        if not faces:
            return None
        
        # Use first face if bbox not specified
        if bbox is None:
            face = faces[0]
        else:
            # Find face closest to provided bbox
            bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            min_dist = float('inf')
            selected_face = None
            
            for face in faces:
                face_center = [(face.bbox[0] + face.bbox[2]) / 2,
                              (face.bbox[1] + face.bbox[3]) / 2]
                dist = np.sqrt((face_center[0] - bbox_center[0])**2 +
                             (face_center[1] - bbox_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    selected_face = face
            
            face = selected_face if selected_face else faces[0]
        
        # Extract embedding
        embedding = face.embedding
        
        return embedding
    
    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Process an image: detect face, preprocess, align, and extract embedding.
        Pipeline: Image → CLAHE → Detection → Quality Filter → Alignment → Embedding
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary containing:
            - embedding: Face embedding vector (512-dimensional)
            - bbox: Bounding box [x1, y1, x2, y2]
            - landmarks: Facial landmarks
            - score: Detection confidence
            - quality_score: Face quality score (if quality filter enabled)
            or None if no face detected or quality too low
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Apply CLAHE preprocessing if enabled
        processed_image = image.copy()
        if self.enable_clahe:
            processed_image = self._apply_clahe(processed_image)
        
        # Detect faces
        faces = self.detect_faces(processed_image)
        
        if not faces:
            return None
        
        # Use first face
        face_info = faces[0]
        
        # Apply quality filtering if enabled
        if self.enable_quality_filter and self.quality_assessor is not None:
            # Extract face region for quality assessment
            x1, y1, x2, y2 = face_info['bbox']
            face_region = processed_image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Assess quality
            quality_metrics = self.quality_assessor.assess_quality(
                face_region,
                face_info['bbox'],
                face_info.get('landmarks'),
                face_info.get('score', 1.0)
            )
            
            # Get quality score (use weighted score for better assessment)
            quality_score = quality_metrics.get('weighted_quality_score', quality_metrics.get('quality_score', 0.0))
            
            # Filter low quality faces
            if quality_score < self.min_quality:
                return None
            
            face_info['quality_score'] = quality_score
        
        # Extract embedding (InsightFace handles alignment internally)
        embedding = self.extract_embedding(processed_image, face_info['bbox'])
        
        if embedding is None:
            return None
        
        result = {
            'embedding': embedding,
            'bbox': face_info['bbox'],
            'landmarks': face_info['landmarks'],
            'score': face_info['score']
        }
        
        if 'quality_score' in face_info:
            result['quality_score'] = face_info['quality_score']
        
        return result
    
    def process_multiple_faces(self, image_path: str) -> List[Dict]:
        """
        Process an image and extract all faces with preprocessing.
        Pipeline: Image → CLAHE → Detection → Quality Filter → Alignment → Embedding
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of face dictionaries, each containing:
            - embedding: Face embedding vector
            - bbox: Bounding box
            - landmarks: Facial landmarks
            - score: Detection confidence
            - quality_score: Face quality score (if quality filter enabled)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Apply CLAHE preprocessing if enabled
        processed_image = image.copy()
        if self.enable_clahe:
            processed_image = self._apply_clahe(processed_image)
        
        # Detect all faces
        faces = self.detect_faces(processed_image)
        
        if not faces:
            return []
        
        results = []
        for face_info in faces:
            # Apply quality filtering if enabled
            if self.enable_quality_filter and self.quality_assessor is not None:
                # Extract face region for quality assessment
                x1, y1, x2, y2 = face_info['bbox']
                face_region = processed_image[y1:y2, x1:x2]
                
                if face_region.size == 0:
                    continue
                
                # Assess quality
                quality_metrics = self.quality_assessor.assess_quality(
                    face_region,
                    face_info['bbox'],
                    face_info.get('landmarks'),
                    face_info.get('score', 1.0)
                )
                
                # Get quality score (use weighted score for better assessment)
                quality_score = quality_metrics.get('weighted_quality_score', quality_metrics.get('quality_score', 0.0))
                
                # Filter low quality faces
                if quality_score < self.min_quality:
                    continue
                
                face_info['quality_score'] = quality_score
            
            # Extract embedding for each face
            embedding = self.extract_embedding(processed_image, face_info['bbox'])
            
            if embedding is not None:
                result = {
                    'embedding': embedding,
                    'bbox': face_info['bbox'],
                    'landmarks': face_info['landmarks'],
                    'score': face_info['score']
                }
                
                if 'quality_score' in face_info:
                    result['quality_score'] = face_info['quality_score']
                
                results.append(result)
        
        return results

