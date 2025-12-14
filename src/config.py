"""
Configuration settings for the face recognition project
"""
from pathlib import Path

class Settings:
    # Directory paths
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    EMBEDDINGS_DIR = "data/embeddings"
    
    # Face detection settings
    DETECTOR = "insightface"  # Options: "insightface" or "mtcnn"
    OUTPUT_SIZE = (112, 112)  # Standard ArcFace input size
    MIN_FACE_WIDTH_PX = 50  # Minimum face width in pixels
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) settings
    CLAHE_CLIP = 2.0
    CLAHE_TILE = (8, 8)
    
    # MLflow tracking
    MLFLOW_TRACKING_URI = "file:./mlflow.db"

# Create a singleton instance
settings = Settings()

