#!/usr/bin/env python3
"""
Face Recognition API
FastAPI backend for face recognition using trained models
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
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
            logger.info(f"✅ Model loaded: {type(model).__name__} from {model_path}")
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
            raise FileNotFoundError("No embeddings directory found")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    logger.info("🚀 Starting Face Recognition API...")
    load_models()
    logger.info("✅ API ready!")

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

@app.get("/classes")
async def get_classes():
    """Get list of recognized classes"""
    return {
        "classes": class_names,
        "count": len(class_names)
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
            "top_predictions": top_predictions[:3],
            "face_bbox": {
                "x": int(face.bbox[0]),
                "y": int(face.bbox[1]),
                "width": int(face.bbox[2] - face.bbox[0]),
                "height": int(face.bbox[3] - face.bbox[1])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/recognize-base64")
async def recognize_face_base64(image_data: dict):
    """Recognize face from base64 encoded image"""
    if model is None or app_face is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Decode base64 image
        base64_string = image_data.get("image")
        if not base64_string:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
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
            "top_predictions": top_predictions[:3],
            "face_bbox": {
                "x": int(face.bbox[0]),
                "y": int(face.bbox[1]),
                "width": int(face.bbox[2] - face.bbox[0]),
                "height": int(face.bbox[3] - face.bbox[1])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/recognize-multiple")
async def recognize_multiple_faces(file: UploadFile = File(...)):
    """Recognize multiple faces in uploaded image"""
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
                "message": "No faces detected in the image",
                "faces_detected": 0,
                "results": []
            }
        
        # Process all faces
        results = []
        start_time = time.time()
        
        for i, face in enumerate(faces):
            embedding = face.embedding
            
            # Make prediction
            probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            predicted_name = class_names[prediction_idx]
            
            # Get top 3 predictions for this face
            top_predictions = []
            for j, prob in enumerate(probabilities):
                top_predictions.append({
                    "name": class_names[j],
                    "confidence": float(prob)
                })
            
            # Sort by confidence
            top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            results.append({
                "face_id": i,
                "prediction": predicted_name,
                "confidence": float(confidence),
                "top_predictions": top_predictions[:3],
                "bbox": {
                    "x": int(face.bbox[0]),
                    "y": int(face.bbox[1]),
                    "width": int(face.bbox[2] - face.bbox[0]),
                    "height": int(face.bbox[3] - face.bbox[1])
                }
            })
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "processing_time": processing_time,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing multiple faces: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
