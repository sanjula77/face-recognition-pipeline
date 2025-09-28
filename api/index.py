#!/usr/bin/env python3
"""
Vercel API Handler for Face Recognition
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
import base64
import io
from PIL import Image
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition using trained models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
app_face = None
class_names = []

def load_models():
    """Load the trained model and face detection app"""
    global model, app_face, class_names
    
    try:
        logger.info("Loading face recognition models...")
        
        # Try to load the production model first
        model_path = Path('production_models/face_recognizer.joblib')
        if not model_path.exists():
            # Fallback to corrected_comparison_results
            model_path = Path('corrected_comparison_results/embeddings_mode_models/logisticregression.joblib')
        
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(model).__name__}")
        else:
            raise FileNotFoundError("No trained model found")
        
        # Initialize InsightFace
        app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app_face.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("✅ InsightFace initialized")
        
        # Load class names from embeddings
        embeddings_dir = Path('data/embeddings')
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob('*.npy'))
            class_names = sorted(list(set([f.stem.split('_')[0] for f in embedding_files if not f.name.startswith('embeddings_database')])))
            logger.info(f"✅ Loaded {len(class_names)} classes: {class_names}")
        else:
            # Fallback class names
            class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']
            logger.info(f"✅ Using fallback classes: {class_names}")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        # Use fallback
        class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']

# Load models on startup
load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Face Recognition API is running!",
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": class_names,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "face_detector_loaded": app_face is not None,
        "num_classes": len(class_names),
        "classes": class_names,
        "model_type": type(model).__name__ if model else None
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize face in uploaded image"""
    if model is None or app_face is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded image
        contents = await file.read()
        
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = app_face.get(img_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No face detected in the image",
                "faces_detected": 0,
                "processing_time": 0
            }
        
        # Process the first face
        face = faces[0]
        embedding = face.embedding
        
        # Make prediction
        start_time = time.time()
        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        predicted_name = class_names[prediction_idx]
        processing_time = time.time() - start_time
        
        # Get top 3 predictions
        top_predictions = []
        for i, prob in enumerate(probabilities):
            top_predictions.append({
                "name": class_names[i],
                "confidence": float(prob)
            })
        
        # Sort by confidence
        top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "prediction": predicted_name,
            "confidence": float(confidence),
            "processing_time": processing_time,
            "faces_detected": len(faces),
            "top_predictions": top_predictions[:3]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Vercel handler
def handler(request):
    """Vercel serverless function handler"""
    return app
