#!/usr/bin/env python3
"""
Simple Face Recognition Inference Script
"""

import joblib
import numpy as np
import cv2
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

def recognize_face(image_path):
    """Recognize face in image"""
    # Load model
    model = joblib.load('models/production/face_recognizer.joblib')
    
    # Initialize detector
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Get class names
    embeddings_dir = Path('data/embeddings')
    embedding_files = list(embeddings_dir.glob('*.npy'))
    class_names = sorted(list(set([f.stem.split('_')[0] for f in embedding_files])))
    
    # Process image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    
    if faces:
        face = faces[0]
        embedding = face.embedding
        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        predicted_name = class_names[prediction_idx]
        
        return predicted_name, confidence
    else:
        return None, 0.0

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        name, confidence = recognize_face(image_path)
        if name:
            print(f"Prediction: {name} ({confidence:.1%})")
        else:
            print("No face detected")
    else:
        print("Usage: python face_recognizer.py <image_path>")
