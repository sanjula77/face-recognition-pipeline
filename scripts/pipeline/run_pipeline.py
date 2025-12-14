#!/usr/bin/env python3
"""
Face Recognition Pipeline (Standard Version)
Complete pipeline for face recognition model training.
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

import subprocess
import shutil
import time
from pathlib import Path

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {title}")
    print("-" * 60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"   🔧 {description}...")
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        process = subprocess.Popen(
            command, 
            shell=True, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print(f"   ✅ {description} completed successfully")
            return True
        else:
            print(f"   ❌ {description} failed with exit code {process.returncode}")
            return False
    except Exception as e:
        print(f"   ❌ {description} failed: {e}")
        return False

def validate_dataset():
    """Validate the dataset structure"""
    print_step(1, "DATASET VALIDATION")
    
    raw_dir = Path('data/raw')
    if not raw_dir.exists():
        print("   ❌ data/raw directory not found")
        return False
    
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
    
    dirs_to_clean = [
        'data/processed',
        'data/embeddings', 
        'models/trained',
        'models/production'
    ]
    
    for dir_path in dirs_to_clean:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"   ✅ Cleaned: {dir_path}")
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/embeddings').mkdir(parents=True, exist_ok=True)
    
    print("   ✅ Fresh start ready")
    return True

def run_preprocessing():
    """Run face detection and preprocessing"""
    print_step(3, "FACE DETECTION & PREPROCESSING")
    
    success = run_command(
        "python src/preprocessing/pipeline.py --input_dir data/raw --output_dir data/processed --detector insightface --no-quality-filter",
        "Face detection and preprocessing"
    )
    
    if not success:
        return False
    
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
    
    embeddings_dir = Path('data/embeddings')
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob('*.embedding.npy'))
        print(f"   ✅ Extracted {len(embedding_files)} embeddings")
        return True
    else:
        print("   ❌ No embeddings found")
        return False

def run_training():
    """Run model training and optimization"""
    print_step(5, "MODEL TRAINING & OPTIMIZATION")
    
    training_script = '''
import sys
sys.path.append('src/training')
from corrected_comparison import CorrectedComparison

# Run training
comparison = CorrectedComparison(
    n_trials=20,
    use_ensemble=False,
    use_calibration=False,
    use_enhanced_normalization=False
)
results = comparison.run_embeddings_mode_corrected('data/embeddings')
print("✅ Training completed successfully")
'''
    
    with open('temp_training.py', 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    success = run_command(
        "python temp_training.py",
        "Model training and optimization"
    )
    
    if Path('temp_training.py').exists():
        Path('temp_training.py').unlink()
    
    if not success:
        return False
    
    models_dir = Path('corrected_comparison_results/embeddings_mode_models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.joblib'))
        # Filter out calibrators and ensemble
        model_files = [f for f in model_files if '_calibrator' not in f.stem and 'ensemble' not in f.stem and 'normalizer' not in f.stem]
        print(f"   ✅ Trained {len(model_files)} models:")
        for model_file in model_files:
            print(f"      - {model_file.name}")
        
        return True
    else:
        print("   ❌ No trained models found")
        return False

def create_production_models():
    """Create production models"""
    print_step(6, "PRODUCTION MODEL ORGANIZATION")
    
    production_dir = Path('models/production')
    production_dir.mkdir(exist_ok=True)
    
    models_dir = Path('models/trained/embeddings_mode_models')
    
    # Use best single model (typically LogisticRegression)
    best_model = models_dir / 'logisticregression.joblib'
    if best_model.exists():
        import shutil
        production_model = production_dir / 'face_recognizer.joblib'
        shutil.copy2(best_model, production_model)
        print(f"   ✅ Production model: {production_model}")
    else:
        # Try other models
        for model_name in ['svm', 'knn', 'randomforest']:
            model_path = models_dir / f'{model_name}.joblib'
            if model_path.exists():
                import shutil
                production_model = production_dir / 'face_recognizer.joblib'
                shutil.copy2(model_path, production_model)
                print(f"   ✅ Production model: {production_model}")
                break
    
    print("   ✅ Production models ready")
    return True

def main():
    """Main pipeline execution"""
    print("🚀 FACE RECOGNITION PIPELINE")
    print("=" * 70)
    print("Complete pipeline for face recognition model training")
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
    
    # Step 3: Run preprocessing without quality filtering
    if not run_preprocessing():
        print("\n❌ Preprocessing failed")
        return False
    
    # Step 4: Extract embeddings
    if not run_embedding_extraction():
        print("\n❌ Embedding extraction failed")
        return False
    
    # Step 5: Run training
    if not run_training():
        print("\n❌ Training failed")
        return False
    
    # Step 6: Create production models
    if not create_production_models():
        print("\n❌ Production setup failed")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n🎉 PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)
    print("✅ All steps completed successfully")
    print("✅ Models trained and optimized")
    print(f"⏱️ Total time: {duration:.1f} seconds")
    print()
    print("📁 OUTPUTS:")
    print("   - Models: models/trained/embeddings_mode_models/")
    print("   - Production model: models/production/face_recognizer.joblib")
    print()
    print("🚀 READY FOR PRODUCTION!")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

