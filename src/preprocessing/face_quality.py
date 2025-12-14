"""
Face Quality Assessment Module
Assesses face quality and filters low-quality faces
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

class FaceQualityAssessor:
    """
    Assesses face quality based on multiple criteria.
    """
    
    def __init__(self, 
                 min_sharpness=50.0,
                 min_brightness=30.0,
                 max_brightness=220.0,
                 min_face_size=50,
                 max_blur_threshold=100.0,
                 min_eye_distance=20):
        """
        Initialize quality assessor.
        
        Args:
            min_sharpness: Minimum sharpness score
            min_brightness: Minimum brightness value
            max_brightness: Maximum brightness value
            min_face_size: Minimum face width/height in pixels
            max_blur_threshold: Maximum blur score (lower is better)
            min_eye_distance: Minimum distance between eyes
        """
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size
        self.max_blur_threshold = max_blur_threshold
        self.min_eye_distance = min_eye_distance
    
    def compute_sharpness(self, img: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance.
        
        Args:
            img: Input image (BGR or grayscale)
        
        Returns:
            Sharpness score (higher is better)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return float(sharpness)
    
    def compute_blur(self, img: np.ndarray) -> float:
        """
        Compute blur score using variance of Laplacian.
        
        Args:
            img: Input image
        
        Returns:
            Blur score (lower is better, 0 = very blurry)
        """
        return self.compute_sharpness(img)
    
    def compute_brightness(self, img: np.ndarray) -> float:
        """
        Compute average brightness.
        
        Args:
            img: Input image
        
        Returns:
            Average brightness (0-255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return float(np.mean(gray))
    
    def compute_contrast(self, img: np.ndarray) -> float:
        """
        Compute image contrast (standard deviation).
        
        Args:
            img: Input image
        
        Returns:
            Contrast score (higher is better)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return float(np.std(gray))
    
    def compute_face_size(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Compute face size from bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            (width, height)
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    
    def compute_eye_distance(self, landmarks: np.ndarray) -> float:
        """
        Compute distance between eyes.
        
        Args:
            landmarks: Face landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        
        Returns:
            Eye distance in pixels
        """
        if landmarks is None or len(landmarks) < 2:
            return 0.0
        
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        return float(distance)
    
    def assess_quality(self, img: np.ndarray, bbox: List[int] = None, 
                      landmarks: np.ndarray = None, detection_score: float = 1.0) -> Dict:
        """
        Assess overall face quality.
        
        Args:
            img: Face image
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Face landmarks
            detection_score: Detection confidence score
        
        Returns:
            Dictionary with quality metrics and overall score
        """
        metrics = {}
        
        # Sharpness/Blur
        sharpness = self.compute_sharpness(img)
        blur = self.compute_blur(img)
        metrics['sharpness'] = sharpness
        metrics['blur'] = blur
        metrics['is_sharp'] = sharpness >= self.min_sharpness
        
        # Brightness
        brightness = self.compute_brightness(img)
        metrics['brightness'] = brightness
        metrics['is_bright_ok'] = self.min_brightness <= brightness <= self.max_brightness
        
        # Contrast
        contrast = self.compute_contrast(img)
        metrics['contrast'] = contrast
        
        # Face size
        if bbox is not None:
            width, height = self.compute_face_size(bbox)
            min_size = min(width, height)
            metrics['face_width'] = width
            metrics['face_height'] = height
            metrics['min_face_size'] = min_size
            metrics['is_size_ok'] = min_size >= self.min_face_size
        else:
            metrics['is_size_ok'] = True  # Assume OK if no bbox
        
        # Eye distance
        if landmarks is not None:
            eye_dist = self.compute_eye_distance(landmarks)
            metrics['eye_distance'] = eye_dist
            metrics['is_eye_dist_ok'] = eye_dist >= self.min_eye_distance
        else:
            metrics['is_eye_dist_ok'] = True  # Assume OK if no landmarks
        
        # Detection score
        metrics['detection_score'] = detection_score
        metrics['is_detection_ok'] = detection_score >= 0.5
        
        # Overall quality score (0-1)
        quality_factors = [
            metrics['is_sharp'],
            metrics['is_bright_ok'],
            metrics['is_size_ok'],
            metrics['is_eye_dist_ok'],
            metrics['is_detection_ok']
        ]
        
        quality_score = sum(quality_factors) / len(quality_factors)
        
        # Weighted quality score (more nuanced)
        weighted_score = (
            0.3 * (1.0 if metrics['is_sharp'] else sharpness / self.min_sharpness) +
            0.2 * (1.0 if metrics['is_bright_ok'] else 0.5) +
            0.2 * (1.0 if metrics['is_size_ok'] else 0.5) +
            0.15 * (1.0 if metrics['is_eye_dist_ok'] else 0.5) +
            0.15 * detection_score
        )
        weighted_score = min(1.0, max(0.0, weighted_score))
        
        metrics['quality_score'] = quality_score
        metrics['weighted_quality_score'] = weighted_score
        metrics['is_high_quality'] = quality_score >= 0.7
        metrics['is_acceptable'] = quality_score >= 0.5
        
        return metrics
    
    def filter_faces(self, faces: List[Dict], min_quality: float = 0.5) -> List[Dict]:
        """
        Filter faces based on quality.
        
        Args:
            faces: List of face dictionaries with 'img', 'bbox', 'landmarks', 'score'
            min_quality: Minimum quality score to keep
        
        Returns:
            Filtered list of faces
        """
        filtered = []
        
        for face in faces:
            img = face.get('img')
            bbox = face.get('bbox')
            landmarks = face.get('landmarks')
            score = face.get('score', 1.0)
            
            if img is None:
                continue
            
            quality = self.assess_quality(img, bbox, landmarks, score)
            
            if quality['weighted_quality_score'] >= min_quality:
                face['quality'] = quality
                filtered.append(face)
            else:
                print(f"   ⚠️  Filtered low-quality face: quality={quality['weighted_quality_score']:.2f}")
        
        return filtered

