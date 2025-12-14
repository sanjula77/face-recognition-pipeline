#!/usr/bin/env python3
"""
Test Model-Based Face Recognition
Tests the trained model on images from data/test/testUsingModel/
Validates predictions against image filenames (expected person names).
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from collections import defaultdict

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Error: InsightFace is not installed!")
    print("   Install with: pip install insightface")
    sys.exit(1)

# Import enhancements
try:
    from src.embeddings.normalization import EmbeddingNormalizer
    from src.training.model_ensemble import ModelEnsemble
    from src.training.confidence_calibration import ConfidenceCalibrator
    HAS_ENHANCEMENTS = True
except ImportError:
    HAS_ENHANCEMENTS = False
    print("⚠️  Enhancement modules not available, using basic model")


def load_model_and_classes(use_ensemble=True, use_calibration=True):
    """Load the trained model and get class names"""
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
                # Try other models
                for model_name in ['svm', 'knn', 'randomforest']:
                    test_path = models_dir / f'{model_name}.joblib'
                    if test_path.exists():
                        model_path = test_path
                        break
        
        if not model_path.exists():
            print("❌ Error: Model not found!")
            print(f"   Checked: {production_dir / 'face_recognizer.joblib'}")
            print(f"   Checked: {models_dir / 'logisticregression.joblib'}")
            sys.exit(1)
        
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path.name} ({type(model).__name__})")
        is_ensemble = False
    
    # Load calibrators if available
    if use_calibration and HAS_ENHANCEMENTS:
        for model_file in models_dir.glob('*_calibrator.joblib'):
            try:
                calibrator = joblib.load(model_file)
                model_name = model_file.stem.replace('_calibrator', '')
                calibrators[model_name] = calibrator
            except Exception:
                pass
    
    # Load normalizer if available
    if HAS_ENHANCEMENTS:
        normalizer_path = models_dir / 'normalizer.joblib'
        if normalizer_path.exists():
            try:
                normalizer = joblib.load(normalizer_path)
                print("✅ Normalizer loaded")
            except Exception:
                pass
    
    # Get class names from embeddings
    embeddings_dir = Path('data/embeddings')
    if not embeddings_dir.exists():
        print("❌ Error: Embeddings directory not found!")
        sys.exit(1)
    
    embedding_files = list(embeddings_dir.glob('*.embedding.npy'))
    if not embedding_files:
        embedding_files = list(embeddings_dir.glob('*.npy'))
    
    if not embedding_files:
        print("❌ Error: No embedding files found!")
        sys.exit(1)
    
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


def extract_face_embedding(image_path, app):
    """Extract face embedding from image"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    
    if not faces:
        return None
    
    # Use first face
    face = faces[0]
    embedding = face.embedding
    
    return embedding


def predict_face(model, embedding, class_names, normalizer=None, calibrators=None, is_ensemble=False):
    """Predict face using model"""
    # Normalize embedding if normalizer available
    if normalizer is not None:
        embedding = normalizer.normalize_single(embedding)
    else:
        # Default L2 normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    # Reshape for model
    embedding = embedding.reshape(1, -1)
    
    # Predict
    if is_ensemble:
        # Ensemble returns probabilities directly
        probabilities = model.predict_proba(embedding)[0]
    else:
        probabilities = model.predict_proba(embedding)[0]
    
    # Apply calibration if available
    if calibrators and not is_ensemble:
        model_name = type(model).__name__.lower()
        if model_name in calibrators:
            probabilities = calibrators[model_name].predict_proba(embedding)[0]
    
    # Get prediction
    prediction_idx = np.argmax(probabilities)
    confidence = probabilities[prediction_idx]
    predicted_name = class_names[prediction_idx]
    
    return predicted_name, confidence, probabilities


def extract_expected_name(filename):
    """Extract expected person name from filename"""
    # Remove extension
    name = Path(filename).stem
    
    # Remove numbers at the end (e.g., "gihan1" -> "gihan")
    while name and name[-1].isdigit():
        name = name[:-1]
    
    return name.lower()


def test_model_recognition(test_dir: str = "data/test/testUsingModel",
                          use_ensemble: bool = True,
                          use_calibration: bool = True,
                          show_details: bool = False):
    """Test model-based recognition on test images"""
    
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    test_images = [f for f in test_path.iterdir() 
                   if f.suffix in image_extensions and f.is_file()]
    
    if not test_images:
        print(f"❌ No test images found in: {test_dir}")
        return False
    
    print("=" * 70)
    print("🧪 MODEL-BASED FACE RECOGNITION TEST")
    print("=" * 70)
    print(f"Test directory: {test_dir}")
    print(f"Found {len(test_images)} test image(s)")
    print()
    
    # Load model and classes
    print("📦 Loading model and classes...")
    model, calibrators, normalizer, is_ensemble, class_names = load_model_and_classes(
        use_ensemble=use_ensemble,
        use_calibration=use_calibration
    )
    print()
    
    # Initialize face detector
    print("🔧 Initializing face detector...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ Face detector ready")
    print()
    
    # Test each image
    print("🔍 Testing images...")
    print("=" * 70)
    
    results = []
    correct_predictions = 0
    total_confidence = 0.0
    
    for img_path in sorted(test_images):
        # Extract expected name from filename
        expected_name = extract_expected_name(img_path.name)
        
        # Extract face embedding
        embedding = extract_face_embedding(img_path, app)
        
        if embedding is None:
            print(f"❌ {img_path.name:30} → No face detected")
            results.append({
                'image': img_path.name,
                'expected': expected_name,
                'predicted': 'No face',
                'confidence': 0.0,
                'correct': False
            })
            continue
        
        # Predict
        predicted_name, confidence, probabilities = predict_face(
            model, embedding, class_names, normalizer, calibrators, is_ensemble
        )
        
        # Check if correct
        is_correct = (predicted_name.lower() == expected_name.lower())
        if is_correct:
            correct_predictions += 1
            total_confidence += confidence
        
        # Store result
        results.append({
            'image': img_path.name,
            'expected': expected_name,
            'predicted': predicted_name,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_path.name:30} → {predicted_name:15} ({confidence:5.1%}) [Expected: {expected_name}]")
        
        if show_details and not is_correct:
            # Show top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            print(f"   Top 3: ", end="")
            for idx in top_indices:
                print(f"{class_names[idx]} ({probabilities[idx]:.1%})", end="  ")
            print()
    
    # Calculate statistics
    total_images = len(results)
    accuracy = correct_predictions / total_images if total_images > 0 else 0.0
    avg_confidence = total_confidence / correct_predictions if correct_predictions > 0 else 0.0
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total images tested: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_images - correct_predictions}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average confidence (correct): {avg_confidence:.1%}")
    print()
    
    # Show incorrect predictions
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print("❌ Incorrect Predictions:")
        for r in incorrect:
            print(f"   {r['image']:30} → Predicted: {r['predicted']:15} (Expected: {r['expected']})")
        print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test model-based face recognition on test images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  python tests/test_model_recognition.py
  
  # Test with custom directory
  python tests/test_model_recognition.py --test_dir data/test/testUsingModel
  
  # Show detailed results
  python tests/test_model_recognition.py --details
        """
    )
    
    parser.add_argument('--test_dir', '-t', type=str, default='data/test/testUsingModel',
                       help='Directory containing test images (default: data/test/testUsingModel)')
    parser.add_argument('--no_ensemble', action='store_true',
                       help='Disable ensemble model (use single model)')
    parser.add_argument('--no_calibration', action='store_true',
                       help='Disable confidence calibration')
    parser.add_argument('--details', '-d', action='store_true',
                       help='Show detailed results for incorrect predictions')
    
    args = parser.parse_args()
    
    success = test_model_recognition(
        test_dir=args.test_dir,
        use_ensemble=not args.no_ensemble,
        use_calibration=not args.no_calibration,
        show_details=args.details
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

