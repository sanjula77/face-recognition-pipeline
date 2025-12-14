"""
One-Shot Learning Face Recognition Module
Face recognition system using one reference image per person.
Uses RetinaFace for detection, ArcFace for embedding, and cosine similarity for matching.
"""

from .database import ReferenceDatabase
from .recognizer import OneShotRecognizer
from .face_processor import FaceProcessor
from .similarity import compute_cosine_similarity, find_best_match

__all__ = [
    'ReferenceDatabase',
    'OneShotRecognizer',
    'FaceProcessor',
    'compute_cosine_similarity',
    'find_best_match'
]

