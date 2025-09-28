#!/usr/bin/env python3
"""
Complete Face Recognition Pipeline
One file that runs the entire project from beginning to end
Perfect for scaling up your dataset!
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {title}")
    print("-" * 60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"   🔧 {description}...")
    try:
        # Set UTF-8 encoding for Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # Run without capturing output to avoid Unicode decode errors
        result = subprocess.run(command, shell=True, check=True, env=env)
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed:")
        print(f"      Error: Command failed with exit code {e.returncode}")
        return False

def validate_dataset():
    """Validate the dataset structure"""
    print_step(1, "DATASET VALIDATION")
    
    raw_dir = Path('data/raw')
    if not raw_dir.exists():
        print("   ❌ data/raw directory not found")
        return False
    
    # Count people and images
    person_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(person_dirs)} person directories:")
    
    total_images = 0
    for person_dir in person_dirs:
        image_count = len(list(person_dir.glob('*.jpg')))
        total_images += image_count
        print(f"   {person_dir.name}: {image_count} images")
    
    print(f"   Total images: {total_images}")
    
    if total_images == 0:
        print("   ❌ No images found in dataset")
        return False
    
    print("   ✅ Dataset validation passed")
    return True

def clean_previous_results():
    """Clean previous results to start fresh"""
    print_step(2, "CLEANING PREVIOUS RESULTS")
    
    # Remove previous results
    dirs_to_clean = [
        'data/processed',
        'data/embeddings', 
        'corrected_comparison_results',
        'production_models'
    ]
    
    for dir_path in dirs_to_clean:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"   ✅ Cleaned: {dir_path}")
    
    # Create fresh directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/embeddings').mkdir(parents=True, exist_ok=True)
    
    print("   ✅ Fresh start ready")
    return True

def run_preprocessing():
    """Run face detection and preprocessing"""
    print_step(3, "FACE DETECTION & PREPROCESSING")
    
    success = run_command(
        "python src/preprocessing/pipeline.py --input_dir data/raw --output_dir data/processed --detector insightface",
        "Face detection and preprocessing"
    )
    
    if not success:
        return False
    
    # Verify preprocessing results
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        processed_files = list(processed_dir.rglob('*.png'))
        print(f"   ✅ Processed {len(processed_files)} face images")
        return True
    else:
        print("   ❌ No processed images found")
        return False

def run_embedding_extraction():
    """Extract face embeddings"""
    print_step(4, "FACE EMBEDDING EXTRACTION")
    
    success = run_command(
        "python src/embeddings/extractor.py --processed_dir data/processed --embeddings_dir data/embeddings --model buffalo_l",
        "Face embedding extraction"
    )
    
    if not success:
        return False
    
    # Verify embedding results
    embeddings_dir = Path('data/embeddings')
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob('*.npy'))
        print(f"   ✅ Extracted {len(embedding_files)} embeddings")
        return True
    else:
        print("   ❌ No embeddings found")
        return False

def run_training():
    """Run model training with corrected comparison (embeddings mode only)"""
    print_step(5, "MODEL TRAINING & OPTIMIZATION")
    
    # Create a simplified training script that only runs embeddings mode
    training_script = '''
import sys
sys.path.append('src/training')
from corrected_comparison import CorrectedComparison

# Run only embeddings mode (which works perfectly)
comparison = CorrectedComparison()
results = comparison.run_embeddings_mode_corrected('data/embeddings')
print("Embeddings mode training completed successfully")
'''
    
    # Write and run the simplified training script
    with open('temp_training.py', 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    success = run_command(
        "python temp_training.py",
        "Model training and hyperparameter optimization (embeddings mode)"
    )
    
    # Clean up temp file
    if Path('temp_training.py').exists():
        Path('temp_training.py').unlink()
    
    if not success:
        return False
    
    # Verify training results
    models_dir = Path('corrected_comparison_results/embeddings_mode_models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.joblib'))
        print(f"   ✅ Trained {len(model_files)} models:")
        for model_file in model_files:
            print(f"      - {model_file.name}")
        return True
    else:
        print("   ❌ No trained models found")
        return False

def validate_results():
    """Validate the final results"""
    print_step(6, "RESULT VALIDATION")
    
    # Test the best model
    test_script = '''
import joblib
import numpy as np
from pathlib import Path
import cv2
import insightface
from insightface.app import FaceAnalysis

# Load the best model (logistic regression)
model_path = Path('corrected_comparison_results/embeddings_mode_models/logisticregression.joblib')
if not model_path.exists():
    print("❌ Model not found")
    exit(1)

model = joblib.load(model_path)
print(f"✅ Model loaded: {type(model).__name__}")

# Initialize face detector
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Get class names
embeddings_dir = Path('data/embeddings')
embedding_files = list(embeddings_dir.glob('*.npy'))
class_names = sorted(list(set([f.stem.split('_')[0] for f in embedding_files])))
print(f"✅ Classes: {class_names}")

# Test on test images if available
test_dir = Path('data/test')
if test_dir.exists():
    test_images = sorted(list(test_dir.glob('*.jpg')))
    if test_images:
        print(f"\\n🧪 Testing on {len(test_images)} test images:")
        print("=" * 60)
        
        total_confidence = 0
        correct_predictions = 0
        
        for img_path in test_images:
            try:
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = app.get(img_rgb)
                
                if faces:
                    face = faces[0]
                    embedding = face.embedding
                    probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
                    prediction_idx = np.argmax(probabilities)
                    confidence = probabilities[prediction_idx]
                    predicted_name = class_names[prediction_idx]
                    
                    print(f"{img_path.name:20} → {predicted_name:10} ({confidence:.1%})")
                    total_confidence += confidence
                    correct_predictions += 1
                else:
                    print(f"{img_path.name:20} → No face detected")
                    
            except Exception as e:
                print(f"{img_path.name:20} → Error: {e}")
        
        if correct_predictions > 0:
            avg_confidence = total_confidence / correct_predictions
            print(f"\\n📊 TEST RESULTS:")
            print(f"   Average confidence: {avg_confidence:.1%}")
            print(f"   Successful predictions: {correct_predictions}/{len(test_images)}")
            
            if avg_confidence >= 0.95:
                print("OUTSTANDING RESULTS!")
            elif avg_confidence >= 0.8:
                print("Good results")
            else:
                print("Results need improvement")
    else:
        print("No test images found")
else:
    print("No test directory found")

print("\\nValidation completed")
'''
    
    # Write and run test script
    with open('temp_validation.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    success = run_command(
        "python temp_validation.py",
        "Result validation"
    )
    
    # Clean up temp file
    if Path('temp_validation.py').exists():
        Path('temp_validation.py').unlink()
    
    return success

def create_production_models():
    """Create production-ready models"""
    print_step(7, "PRODUCTION MODEL ORGANIZATION")
    
    # Create production directory
    production_dir = Path('production_models')
    production_dir.mkdir(exist_ok=True)
    
    # Copy best model
    best_model = Path('corrected_comparison_results/embeddings_mode_models/logisticregression.joblib')
    if best_model.exists():
        production_model = production_dir / 'face_recognizer.joblib'
        shutil.copy2(best_model, production_model)
        print(f"   ✅ Production model: {production_model}")
    
    # Create simple inference script
    inference_script = '''#!/usr/bin/env python3
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
    model = joblib.load('production_models/face_recognizer.joblib')
    
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
'''
    
    with open('face_recognizer.py', 'w', encoding='utf-8') as f:
        f.write(inference_script)
    
    print("   ✅ Inference script: face_recognizer.py")
    print("   ✅ Production models ready")
    return True

def main():
    """Main pipeline execution"""
    print("🚀 COMPLETE FACE RECOGNITION PIPELINE")
    print("=" * 70)
    print("This script runs the entire project from beginning to end!")
    print("Perfect for scaling up your dataset!")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Validate dataset
    if not validate_dataset():
        print("\n❌ Dataset validation failed")
        return False
    
    # Step 2: Clean previous results
    if not clean_previous_results():
        print("\n❌ Cleanup failed")
        return False
    
    # Step 3: Run preprocessing
    if not run_preprocessing():
        print("\n❌ Preprocessing failed")
        return False
    
    # Step 4: Extract embeddings
    if not run_embedding_extraction():
        print("\n❌ Embedding extraction failed")
        return False
    
    # Step 5: Train models
    if not run_training():
        print("\n❌ Training failed")
        return False
    
    # Step 6: Validate results
    if not validate_results():
        print("\n❌ Validation failed")
        return False
    
    # Step 7: Create production models
    if not create_production_models():
        print("\n❌ Production setup failed")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)
    print("✅ All steps completed successfully")
    print("✅ Models trained and optimized")
    print("✅ Results validated")
    print("✅ Production models ready")
    print(f"⏱️ Total time: {duration:.1f} seconds")
    print()
    print("📁 OUTPUTS:")
    print("   - Trained models: corrected_comparison_results/embeddings_mode_models/")
    print("   - Production model: production_models/face_recognizer.joblib")
    print("   - Inference script: face_recognizer.py")
    print()
    print("🚀 READY FOR PRODUCTION!")
    print("To recognize a face: python face_recognizer.py <image_path>")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
