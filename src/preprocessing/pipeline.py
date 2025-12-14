# src/preprocessing/pipeline.py
import os
import sys

# Fix OpenMP library conflict on Windows
# This is needed when multiple libraries (NumPy, SciPy, InsightFace) use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
from pathlib import Path
import shutil
import json
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.config import settings
from src.preprocessing.detect_align import (
    HAS_INSIGHTFACE, HAS_MTCNN,
    init_insightface_detector, init_mtcnn,
    detect_faces_insightface, detect_faces_mtcnn,
    align_face, apply_clahe, compute_sharpness, compute_brightness,
    normalize_for_arcface, save_normalized_face
)
from src.preprocessing.face_quality import FaceQualityAssessor

def process_image(path, detector_impl, output_size, clahe_clip, clahe_tile, min_face_width):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if detector_impl == 'insightface':
        faces = detect_faces_insightface(process_image.app, img, min_face_width)
    else:
        faces = detect_faces_mtcnn(process_image.app, img, min_face_width)

    results = []
    for idx, f in enumerate(faces):
        lm = f['landmarks']
        if lm is None:
            continue
        # align - expecting 5 points (left_eye,right_eye,nose,left_mouth,right_mouth)
        try:
            aligned = align_face(img, lm, output_size=output_size)
        except Exception as e:
            print("Alignment failed:", e)
            continue
        # CLAHE
        clahe_img = apply_clahe(aligned, clip=clahe_clip, tile=clahe_tile)
        sharp = compute_sharpness(clahe_img)
        bright = compute_brightness(clahe_img)
        
        # Normalize for ArcFace
        normalized_face = normalize_for_arcface(clahe_img)
        
        out = {
            'aligned': clahe_img,
            'normalized': normalized_face,
            'bbox': f['bbox'],
            'score': f['score'],
            'sharpness': sharp,
            'brightness': bright
        }
        results.append(out)
    return results

def run_pipeline(input_dir, output_dir, detector_choice='insightface', ctx_id=0, 
                 enable_quality_filter=True, min_quality=0.5):
    # Initialize MLflow if available
    use_mlflow = HAS_MLFLOW
    mlflow_context = None
    
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            run_name = f"preproc_run_{int(time.time())}"
            mlflow_context = mlflow.start_run(run_name=run_name)
            mlflow.log_param("detector", detector_choice)
            mlflow.log_param("output_size", f"{settings.OUTPUT_SIZE}")
        except Exception:
            use_mlflow = False
            mlflow_context = None
    
    total = 0
    detected = 0
    filtered = 0
    sum_sharp = 0.0
    sum_bright = 0.0

    # Initialize face quality assessor
    quality_assessor = FaceQualityAssessor() if enable_quality_filter else None
    if enable_quality_filter:
        print("✅ Face quality filtering enabled")

    # initialize detector globally for speed
    if detector_choice == 'insightface':
        print("Initializing insightface detector (RetinaFace)")
        process_image.app = init_insightface_detector(ctx_id=ctx_id)
    else:
        device = 'cuda' if (HAS_MTCNN and getattr(process_image, 'use_cuda', False)) else 'cpu'
        print("Initializing MTCNN on device:", device)
        process_image.app = init_mtcnn(device=device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    
    # Process images in subdirectories (organized by person)
    person_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"\n📁 Found {len(person_dirs)} person directories to process")
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        print(f"\n👤 Processing person: {person_name}")
        
        # Create output directory for this person
        person_output_dir = Path(output_dir) / person_name
        person_output_dir.mkdir(parents=True, exist_ok=True)
        
        person_images = sorted(list(person_dir.glob("*.jpg")))
        print(f"   Found {len(person_images)} images for {person_name}")
        
        person_detected = 0
        for img_path in person_images:
            total += 1
            print(f"   Processing: {img_path.name}", end=" ... ", flush=True)
            try:
                res = process_image(img_path, detector_choice, settings.OUTPUT_SIZE, settings.CLAHE_CLIP, settings.CLAHE_TILE, settings.MIN_FACE_WIDTH_PX)
                if res is None or len(res) == 0:
                    print("No face detected")
                    continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            
            # Apply quality filtering if enabled
            if enable_quality_filter and quality_assessor:
                # Prepare faces for quality assessment
                faces_for_quality = []
                for face_data in res:
                    faces_for_quality.append({
                        'img': face_data['aligned'],
                        'bbox': face_data['bbox'],
                        'landmarks': None,  # Could extract from detection if available
                        'score': face_data['score']
                    })
                
                # Filter by quality
                filtered_faces = quality_assessor.filter_faces(faces_for_quality, min_quality=min_quality)
                
                if len(filtered_faces) < len(res):
                    filtered += (len(res) - len(filtered_faces))
                    print(f"   Filtered {len(res) - len(filtered_faces)} low-quality face(s)")
                
                # Reconstruct res with quality info
                if len(filtered_faces) == 0:
                    print("   All faces filtered out (low quality)")
                    continue
                
                # Update res with filtered faces (keep original structure)
                quality_filtered_res = []
                for face_data, quality_face in zip(res, filtered_faces):
                    if 'quality' in quality_face:
                        face_data['quality'] = quality_face['quality']
                        quality_filtered_res.append(face_data)
                
                res = quality_filtered_res
            
            # Save all face crops for this image
            for i, face in enumerate(res):
                # Save PNG preview (for visualization)
                png_fn = f"{img_path.stem}_face{i}.png"
                png_path = person_output_dir / png_fn
                cv2.imwrite(str(png_path), face['aligned'])
                
                # Save NPY file (for ArcFace)
                npy_fn = f"{img_path.stem}_face{i}.npy"
                npy_path = person_output_dir / npy_fn
                save_normalized_face(face['normalized'], str(npy_path))
                
                # Save metadata
                meta = {
                    'source_image': str(img_path),
                    'bbox': face['bbox'],
                    'score': face['score'],
                    'sharpness': face['sharpness'],
                    'brightness': face['brightness'],
                    'normalized_shape': face['normalized'].shape,
                    'normalized_dtype': str(face['normalized'].dtype),
                    'normalized_range': [float(face['normalized'].min()), float(face['normalized'].max())]
                }
                # metadata alongside
                with open(str(png_path) + ".meta.json", "w", encoding="utf8") as f:
                    json.dump(meta, f, indent=2)
                
                # log first processed image as artifact in MLflow for quick visual check
                if detected == 0 and use_mlflow:
                    try:
                        mlflow.log_artifact(str(png_path), artifact_path="sample_processed")
                        mlflow.log_artifact(str(npy_path), artifact_path="sample_normalized")
                    except Exception as e:
                        print(f"Warning: Could not log artifact to MLflow: {e}")
                        print("Continuing without artifact logging...")
                detected += 1
                person_detected += 1
                sum_sharp += face['sharpness']
                sum_bright += face['brightness']
        
        print(f"   ✅ {person_name}: {person_detected} faces detected from {len(person_images)} images")

    # Calculate final statistics after processing all people
    detection_rate = detected / max(total, 1)
    avg_sharp = sum_sharp / max(detected, 1)
    avg_bright = sum_bright / max(detected, 1)
    
    if use_mlflow:
        try:
            mlflow.log_metric("total_images", total)
            mlflow.log_metric("detected_faces", detected)
            mlflow.log_metric("detection_rate", detection_rate)
            mlflow.log_metric("avg_sharpness", avg_sharp)
            mlflow.log_metric("avg_brightness", avg_bright)
        except Exception:
            pass

    print(f"\n✅ PREPROCESSING COMPLETE")
    print(f"   Total images processed: {total}")
    print(f"   Total faces detected: {detected}")
    if enable_quality_filter:
        print(f"   Faces filtered (low quality): {filtered}")
    print(f"   Detection rate: {detection_rate:.1%}")
    print(f"   Processed {len(person_dirs)} people")
    
    # Close MLflow run if it was opened
    if mlflow_context:
        try:
            mlflow.end_run()
        except Exception:
            pass
    
    return {
        'total': total,
        'detected': detected,
        'detection_rate': detection_rate,
        'avg_sharpness': avg_sharp,
        'avg_brightness': avg_bright
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=settings.RAW_DIR)
    parser.add_argument("--output_dir", default=settings.PROCESSED_DIR)
    parser.add_argument("--detector", default=settings.DETECTOR)
    parser.add_argument("--ctx_id", type=int, default=0)  # 0 GPU, -1 CPU for insightface
    parser.add_argument("--quality-filter", action='store_true', default=True,
                       help='Enable face quality filtering (default: True)')
    parser.add_argument("--no-quality-filter", action='store_false', dest='quality_filter',
                       help='Disable face quality filtering')
    parser.add_argument("--min-quality", type=float, default=0.5,
                       help='Minimum quality score to keep face (default: 0.5)')
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.detector, args.ctx_id,
                 enable_quality_filter=args.quality_filter, min_quality=args.min_quality)
