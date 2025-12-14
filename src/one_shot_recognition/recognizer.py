"""
One-Shot Recognizer Module
Performs face recognition using one-shot learning approach.
"""

import numpy as np
from typing import Dict, List, Optional
from .database import ReferenceDatabase
from .face_processor import FaceProcessor
from .similarity import find_best_match, compute_cosine_similarity


class OneShotRecognizer:
    """
    Face recognizer using one-shot learning approach.
    Compares input face embeddings with reference embeddings using cosine similarity.
    """
    
    def __init__(self, database_path: str = "databases/reference_database",
                 similarity_threshold: float = 0.6,
                 model_name: str = 'buffalo_l',
                 ctx_id: int = -1):
        """
        Initialize one-shot recognizer.
        
        Args:
            database_path: Path to reference database
            similarity_threshold: Minimum similarity score to consider a match (0-1)
            model_name: InsightFace model name
            ctx_id: Device ID (-1 for CPU, 0+ for GPU)
        """
        self.database = ReferenceDatabase(database_path)
        self.similarity_threshold = similarity_threshold
        self.face_processor = FaceProcessor(model_name=model_name, ctx_id=ctx_id)
    
    def recognize(self, embedding: np.ndarray, top_k: int = 1) -> List[Dict]:
        """
        Recognize a face by comparing embedding with references.
        
        Args:
            embedding: Face embedding vector (512-dimensional)
            top_k: Number of top matches to return
        
        Returns:
            List of recognition results, each containing:
            - name: Person's name (or 'Unknown')
            - similarity: Similarity score (0-1)
            - metadata: Reference metadata
            - is_match: Whether similarity meets threshold
        """
        if self.database.embeddings is None or len(self.database.embeddings) == 0:
            return [{
                'name': 'Unknown',
                'similarity': 0.0,
                'metadata': None,
                'is_match': False,
                'reason': 'No references in database'
            }]
        
        # Get all references
        reference_embeddings, metadata = self.database.get_all_references()
        reference_names = [m['name'] for m in metadata]
        
        # Find best match
        matches = find_best_match(
            embedding,
            reference_embeddings,
            reference_names,
            threshold=self.similarity_threshold,
            top_k=top_k
        )
        
        # Add metadata to results
        for match in matches:
            if match['index'] >= 0:
                match['metadata'] = metadata[match['index']]
            else:
                match['metadata'] = None
        
        return matches
    
    def recognize_from_image(self, image_path: str, top_k: int = 1) -> Optional[List[Dict]]:
        """
        Recognize face from image file.
        
        Args:
            image_path: Path to image file
            top_k: Number of top matches to return
        
        Returns:
            List of recognition results or None if face not detected
        """
        # Process image to extract face
        result = self.face_processor.process_image(image_path)
        
        if result is None:
            return None
        
        # Recognize using embedding
        return self.recognize(result['embedding'], top_k=top_k)
    
    def recognize_multiple_faces(self, image_path: str) -> List[Dict]:
        """
        Recognize all faces in an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of recognition results for each face, each containing:
            - embedding: Face embedding
            - bbox: Bounding box
            - recognition: Recognition results
        """
        # Process image to extract all faces
        faces = self.face_processor.process_multiple_faces(image_path)
        
        if not faces:
            return []
        
        results = []
        for face_info in faces:
            # Recognize each face
            recognition = self.recognize(face_info['embedding'], top_k=1)
            
            results.append({
                'embedding': face_info['embedding'],
                'bbox': face_info['bbox'],
                'landmarks': face_info.get('landmarks'),
                'score': face_info.get('score'),
                'recognition': recognition[0] if recognition else None
            })
        
        return results
    
    def update_threshold(self, threshold: float) -> None:
        """
        Update similarity threshold.
        
        Args:
            threshold: New similarity threshold (0-1)
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            print(f"✅ Updated similarity threshold to {threshold:.2f}")
        else:
            raise ValueError("Threshold must be between 0 and 1")
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return self.database.get_statistics()

