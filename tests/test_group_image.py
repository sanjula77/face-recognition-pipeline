#!/usr/bin/env python3
"""
Test Group Image Face Recognition
Input: Image path with multiple faces
Output: Annotated image with bounding boxes, names, and confidence scores
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
                is_ensemble = False
    
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
        is_ensemble = False
    
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

def recognize_group_image(image_path, model, class_names, app, output_path=None, 
                         min_confidence=0.3, calibrators=None, normalizer=None):
    """
    Recognize all faces in a group image and draw annotations
    
    Args:
        image_path: Path to the group image
        model: Trained classifier model
        class_names: List of person names
        app: InsightFace FaceAnalysis app
        output_path: Path to save annotated image (optional)
        min_confidence: Minimum confidence to show label
    
    Returns:
        List of recognition results and annotated image
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return [], None
    
    original_img = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect all faces
    print(f"🔍 Detecting faces in image...")
    faces = app.get(img_rgb)
    
    if not faces:
        print("❌ No faces detected in the image")
        return [], original_img
    
    print(f"✅ Detected {len(faces)} face(s)")
    
    results = []
    img_annotated = img.copy()
    
    # Process each face
    for i, face in enumerate(faces, 1):
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
            # If it's an ensemble, use its method
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
            'face_id': i,
            'name': predicted_name,
            'confidence': confidence,
            'bbox': bbox,
            'probabilities': probabilities
        })
        
        # Draw on image
        x1, y1, x2, y2 = bbox
        
        # Choose color based on confidence
        if confidence >= 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence >= 0.5:
            color = (0, 165, 255)  # Orange - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        if confidence >= min_confidence:
            label = f"{predicted_name} ({confidence:.1%})"
        else:
            label = f"Unknown ({confidence:.1%})"
        
        # Calculate label position
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y2 + 25
        
        # Draw label background
        cv2.rectangle(img_annotated,
                     (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0] + 10, label_y + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(img_annotated, label, (x1 + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Print result
        print(f"\n👤 Face {i}:")
        print(f"   Name: {predicted_name}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Location: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Show top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        print(f"   Top 3 Predictions:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"      {rank}. {class_names[idx]}: {probabilities[idx]:.1%}")
    
    # Save annotated image if output path is provided
    if output_path:
        cv2.imwrite(str(output_path), img_annotated)
        print(f"\n💾 Annotated image saved to: {output_path}")
    
    return results, img_annotated

def main():
    parser = argparse.ArgumentParser(description='Test face recognition on a group image')
    parser.add_argument('image_path', type=str, help='Path to the group image file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Path to save annotated image (default: <image_name>_annotated.jpg)')
    parser.add_argument('--show', action='store_true', help='Display annotated image')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                       help='Minimum confidence to show label (default: 0.3)')
    parser.add_argument('--ctx_id', type=int, default=-1, help='Device ID (-1 for CPU, 0 for GPU)')
    parser.add_argument('--use-ensemble', action='store_true', default=True,
                       help='Use ensemble model if available (default: True)')
    parser.add_argument('--use-calibration', action='store_true', default=True,
                       help='Use confidence calibration if available (default: True)')
    
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
                print(f"   - Use full path: python test_group_image.py C:/path/to/image.jpg")
                print(f"   - Or place image in current directory: {Path.cwd()}")
                print(f"   - Current directory: {Path.cwd()}")
                sys.exit(1)
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"\n💡 Tips:")
        print(f"   - Use full path: python test_group_image.py C:/path/to/image.jpg")
        print(f"   - Or place image in current directory")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("👥 FACE RECOGNITION - GROUP IMAGE TEST")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load model and classes with enhancements
    model, calibrators, normalizer, is_ensemble, class_names = load_model_and_classes(
        use_ensemble=True,  # Always try ensemble
        use_calibration=True  # Always try calibration
    )
    
    # Initialize face detector
    print("\n🔧 Initializing face detector...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=args.ctx_id, det_size=(640, 640))
    print("✅ Face detector ready")
    
    # Recognize all faces
    print(f"\n🔍 Processing group image: {image_path.name}")
    results, annotated_img = recognize_group_image(
        image_path, model, class_names, app, 
        output_path=output_path,
        min_confidence=args.min_confidence,
        calibrators=calibrators,
        normalizer=normalizer
    )
    
    if results:
        print(f"\n✅ Successfully recognized {len(results)} face(s)")
        print(f"\n📊 Summary:")
        for result in results:
            print(f"   Face {result['face_id']}: {result['name']} ({result['confidence']:.1%})")
        
        # Display image if requested
        if args.show:
            cv2.imshow('Group Face Recognition Result', annotated_img)
            print("\n📸 Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("\n❌ No faces recognized")

if __name__ == '__main__':
    main()

