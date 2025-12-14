#!/usr/bin/env python3
"""
Test Single Image Face Recognition
Input: Image path
Output: Predicted name and confidence
"""

import os
import sys
from pathlib import Path

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import joblib
import numpy as np
import cv2
import argparse
from sklearn.preprocessing import normalize

# Import enhancements
from src.embeddings.normalization import EmbeddingNormalizer
from src.training.model_ensemble import ModelEnsemble
from src.training.confidence_calibration import ConfidenceCalibrator

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Error: InsightFace is not installed!")
    print("   Install with: pip install insightface")
    sys.exit(1)

def load_model_and_classes(use_ensemble=True, use_calibration=True):
    """Load the trained model (or ensemble) and get class names"""
    models_dir = Path('models/trained/embeddings_mode_models')
    production_dir = Path('models/production')
    
    model = None
    calibrators = {}
    normalizer = None
    is_ensemble = False
    
    # Try to load ensemble first
    if use_ensemble:
        ensemble_path = production_dir / 'face_recognizer_ensemble.joblib'
        if not ensemble_path.exists():
            ensemble_path = models_dir / 'ensemble.joblib'
        
        if ensemble_path.exists():
            try:
                model = joblib.load(ensemble_path)
                print(f"✅ Ensemble model loaded: {ensemble_path.name}")
                is_ensemble = True
            except Exception as e:
                print(f"⚠️  Failed to load ensemble: {e}")
    
    # Load single model if ensemble not available
    if model is None:
        model_path = production_dir / 'face_recognizer.joblib'
        if not model_path.exists():
            model_path = models_dir / 'logisticregression.joblib'
            if not model_path.exists():
                print("❌ Error: Model not found!")
                sys.exit(1)
        
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(model).__name__}")
    
    # Load calibrators if available
    if use_calibration:
        for model_file in models_dir.glob('*_calibrator.joblib'):
            try:
                calibrator = joblib.load(model_file)
                model_name = model_file.stem.replace('_calibrator', '')
                calibrators[model_name] = calibrator
                print(f"✅ Calibrator loaded: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load calibrator {model_file.name}: {e}")
    
    # Load normalizer if available
    normalizer_path = models_dir / 'normalizer.joblib'
    if normalizer_path.exists():
        try:
            normalizer = joblib.load(normalizer_path)
            print(f"✅ Normalizer loaded")
        except:
            pass
    
    # Get class names from embeddings
    embeddings_dir = Path('data/embeddings')
    if not embeddings_dir.exists():
        print("❌ Error: Embeddings directory not found!")
        sys.exit(1)
    
    embedding_files = list(embeddings_dir.glob('*.embedding.npy'))
    if not embedding_files:
        # Try alternative pattern
        embedding_files = list(embeddings_dir.glob('*.npy'))
    
    if not embedding_files:
        print("❌ Error: No embedding files found!")
        sys.exit(1)
    
    # Extract person names from filenames
    class_names = []
    for emb_file in embedding_files:
        filename = emb_file.stem.replace('.embedding', '')
        if '_face' in filename:
            person_name = filename.split('_face')[0].split('_')[0]
        else:
            person_name = filename.split('_')[0]
        if person_name not in class_names:
            class_names.append(person_name)
    
    class_names = sorted(class_names)
    print(f"✅ Found {len(class_names)} classes: {class_names}")
    
    return model, calibrators, normalizer, is_ensemble, class_names

def recognize_face(image_path, model, class_names, app, show_image=False,
                  calibrators=None, normalizer=None):
    """
    Recognize face in a single image
    
    Args:
        image_path: Path to the image
        model: Trained classifier model
        class_names: List of person names
        app: InsightFace FaceAnalysis app
        show_image: Whether to display the image with annotation
    
    Returns:
        List of (name, confidence, bbox) for each detected face
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return []
    
    # Convert to RGB for InsightFace
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = app.get(img_rgb)
    
    if not faces:
        print("❌ No faces detected in the image")
        return []
    
    results = []
    
    for face in faces:
        # Get embedding
        embedding = face.embedding
        
        # Enhanced normalization
        if normalizer is not None:
            embedding_norm = normalizer.normalize_single(embedding)
        else:
            embedding_norm = normalize(embedding.reshape(1, -1), axis=1)[0]
        
        # Predict using ensemble or single model
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(embedding_norm.reshape(1, -1))[0]
        else:
            probabilities = model.predict_proba(embedding_norm.reshape(1, -1))[0]
        
        # Apply calibration if available
        if calibrators:
            # Use first available calibrator
            calibrator = list(calibrators.values())[0]
            probabilities = calibrator.calibrate_proba(probabilities.reshape(1, -1))[0]
        
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        predicted_name = class_names[prediction_idx]
        
        # Get bounding box
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        
        results.append({
            'name': predicted_name,
            'confidence': confidence,
            'bbox': bbox,
            'probabilities': probabilities
        })
        
        # Print result
        print(f"\n👤 Face Detected:")
        print(f"   Name: {predicted_name}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
        
        # Show top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        print(f"\n   Top 3 Predictions:")
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i}. {class_names[idx]}: {probabilities[idx]:.1%}")
    
    # Display image with annotations if requested
    if show_image and results:
        img_display = img.copy()
        for result in results:
            bbox = result['bbox']
            name = result['name']
            conf = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{name} ({conf:.1%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[3] + 25
            
            # Draw label background
            cv2.rectangle(img_display, 
                         (bbox[0], label_y - label_size[1] - 5),
                         (bbox[0] + label_size[0], label_y + 5),
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img_display, label, (bbox[0], label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Display image
        cv2.imshow('Face Recognition Result', img_display)
        print("\n📸 Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test face recognition on a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Display image with annotations')
    parser.add_argument('--model', type=str, default=None, help='Path to model file (optional)')
    parser.add_argument('--ctx_id', type=int, default=-1, help='Device ID (-1 for CPU, 0 for GPU)')
    
    args = parser.parse_args()
    
    # Check if image exists (try current directory and absolute path)
    image_path = Path(args.image_path)
    
    # If not absolute path, try current directory first
    if not image_path.is_absolute():
        if not image_path.exists():
            # Try in current working directory
            cwd_path = Path.cwd() / image_path
            if cwd_path.exists():
                image_path = cwd_path
            else:
                print(f"❌ Error: Image not found: {args.image_path}")
                print(f"   Tried: {Path.cwd() / args.image_path}")
                print(f"   Tried: {Path(args.image_path).resolve()}")
                print(f"\n💡 Tips:")
                print(f"   - Use full path: python test_single_image.py C:/path/to/image.jpg")
                print(f"   - Or place image in current directory: {Path.cwd()}")
                print(f"   - Current directory: {Path.cwd()}")
                sys.exit(1)
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"\n💡 Tips:")
        print(f"   - Use full path: python test_single_image.py C:/path/to/image.jpg")
        print(f"   - Or place image in current directory")
        sys.exit(1)
    
    print("=" * 60)
    print("🔍 FACE RECOGNITION - SINGLE IMAGE TEST")
    print("=" * 60)
    print(f"Image: {image_path}")
    print()
    
    # Load model and classes with enhancements
    model, calibrators, normalizer, is_ensemble, class_names = load_model_and_classes(
        use_ensemble=True,
        use_calibration=True
    )
    
    # Initialize face detector
    print("\n🔧 Initializing face detector...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=args.ctx_id, det_size=(640, 640))
    print("✅ Face detector ready")
    
    # Recognize faces
    print(f"\n🔍 Processing image: {image_path.name}")
    results = recognize_face(image_path, model, class_names, app, show_image=args.show,
                            calibrators=calibrators, normalizer=normalizer)
    
    if results:
        print(f"\n✅ Successfully recognized {len(results)} face(s)")
    else:
        print("\n❌ No faces recognized")

if __name__ == '__main__':
    main()

