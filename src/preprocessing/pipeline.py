# src/preprocessing/pipeline.py
import os
import sys
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
import mlflow

from src.config import settings
from src.preprocessing.detect_align import (
    HAS_INSIGHTFACE, HAS_MTCNN,
    init_insightface_detector, init_mtcnn,
    detect_faces_insightface, detect_faces_mtcnn,
    align_face, apply_clahe, compute_sharpness, compute_brightness,
    normalize_for_arcface, save_normalized_face
)

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

def run_pipeline(input_dir, output_dir, detector_choice='insightface', ctx_id=0):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    run_name = f"preproc_run_{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("detector", detector_choice)
        mlflow.log_param("output_size", f"{settings.OUTPUT_SIZE}")
        total = 0
        detected = 0
        sum_sharp = 0.0
        sum_bright = 0.0

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
        for person_dir in sorted(input_dir.iterdir()):
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            print(f"Processing person: {person_name}")
            
            # Create output directory for this person
            person_output_dir = Path(output_dir) / person_name
            person_output_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in sorted(person_dir.glob("*.jpg")):
                total += 1
                print("Processing:", img_path.name)
                res = process_image(img_path, detector_choice, settings.OUTPUT_SIZE, settings.CLAHE_CLIP, settings.CLAHE_TILE, settings.MIN_FACE_WIDTH_PX)
                if res is None or len(res) == 0:
                    print(" - No face detected")
                    continue
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
                if detected == 0:
                    try:
                        mlflow.log_artifact(str(png_path), artifact_path="sample_processed")
                        mlflow.log_artifact(str(npy_path), artifact_path="sample_normalized")
                    except Exception as e:
                        print(f"Warning: Could not log artifact to MLflow: {e}")
                        print("Continuing without artifact logging...")
                detected += 1
                sum_sharp += face['sharpness']
                sum_bright += face['brightness']

        detection_rate = detected / max(total, 1)
        avg_sharp = sum_sharp / max(detected, 1)
        avg_bright = sum_bright / max(detected, 1)
        mlflow.log_metric("total_images", total)
        mlflow.log_metric("detected_faces", detected)
        mlflow.log_metric("detection_rate", detection_rate)
        mlflow.log_metric("avg_sharpness", avg_sharp)
        mlflow.log_metric("avg_brightness", avg_bright)

        print("Done. Total:", total, "Detected faces:", detected)
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
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.detector, args.ctx_id)
