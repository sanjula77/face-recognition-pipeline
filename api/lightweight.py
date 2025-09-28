#!/usr/bin/env python3
"""
Lightweight Face Recognition API for Vercel
Uses smaller models and optimized code
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
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
    title="Lightweight Face Recognition API",
    description="Optimized API for Vercel deployment",
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

# Fallback class names (since we can't load large models on Vercel)
class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Lightweight Face Recognition API is running!",
        "status": "healthy",
        "note": "This is a lightweight version for Vercel deployment",
        "classes": class_names,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_type": "lightweight",
        "num_classes": len(class_names),
        "classes": class_names,
        "note": "Full models not available on Vercel due to size limits"
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Lightweight face recognition (demo version)"""
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
        
        # Simple face detection using OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected in the image",
                "faces_detected": 0,
                "processing_time": 0,
                "note": "This is a demo version - full recognition requires larger models"
            }
        
        # Simulate recognition (since we can't load large models on Vercel)
        start_time = time.time()
        
        # Simple mock prediction based on image properties
        img_hash = hash(contents) % len(class_names)
        predicted_name = class_names[img_hash]
        confidence = 0.75 + (img_hash % 25) / 100  # Mock confidence between 0.75-1.0
        
        processing_time = time.time() - start_time
        
        # Mock top predictions
        top_predictions = []
        for i, name in enumerate(class_names):
            if i == img_hash:
                top_predictions.append({"name": name, "confidence": confidence})
            else:
                mock_conf = (1 - confidence) / (len(class_names) - 1)
                top_predictions.append({"name": name, "confidence": mock_conf})
        
        top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "prediction": predicted_name,
            "confidence": confidence,
            "processing_time": processing_time,
            "faces_detected": len(faces),
            "top_predictions": top_predictions[:3],
            "note": "This is a demo version - full recognition requires deployment on platforms with larger size limits"
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
