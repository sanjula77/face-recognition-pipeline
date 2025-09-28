#!/usr/bin/env python3
"""
Minimal Face Recognition API for Vercel
Ultra-lightweight version that fits Vercel's limits
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API (Vercel)",
    description="Lightweight face recognition for Vercel deployment",
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

# Class names
class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Face Recognition API (Vercel) is running!",
        "status": "healthy",
        "version": "1.0.0",
        "note": "This is a lightweight version for Vercel deployment"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "api_type": "vercel-lightweight",
        "classes": class_names,
        "note": "For full functionality, deploy to Google Cloud Run"
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Lightweight face recognition"""
    try:
        # Read image
        contents = await file.read()
        
        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Simple face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected",
                "faces_detected": 0,
                "note": "This is a demo version - full recognition requires Google Cloud Run"
            }
        
        # Mock prediction
        start_time = time.time()
        img_hash = hash(contents) % len(class_names)
        predicted_name = class_names[img_hash]
        confidence = 0.75 + (img_hash % 25) / 100
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "prediction": predicted_name,
            "confidence": confidence,
            "processing_time": processing_time,
            "faces_detected": len(faces),
            "note": "Demo version - deploy to Google Cloud Run for full recognition"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Vercel handler
def handler(request):
    return app
