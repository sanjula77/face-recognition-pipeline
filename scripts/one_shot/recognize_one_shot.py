#!/usr/bin/env python3
"""
One-Shot Face Recognition
Recognizes faces using one-shot learning approach (one reference image per person).
Uses RetinaFace for detection, ArcFace for embedding, and cosine similarity for matching.
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

import argparse
import cv2
import numpy as np
from src.one_shot_recognition.recognizer import OneShotRecognizer


def recognize_single_face(image_path: str,
                          database_path: str = "databases/reference_database",
                          similarity_threshold: float = 0.6,
                          top_k: int = 3,
                          show_image: bool = False) -> bool:
    """
    Recognize a single face in an image.
    
    Args:
        image_path: Path to image file
        database_path: Path to reference database
        similarity_threshold: Minimum similarity for a match
        top_k: Number of top matches to show
        show_image: Whether to display the image
    
    Returns:
        True if successful, False otherwise
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print("=" * 70)
    print("🔍 ONE-SHOT FACE RECOGNITION")
    print("=" * 70)
    print("System: Image → Preprocessing (CLAHE) → Face Detection (RetinaFace) →")
    print("        Quality Filter → Face Alignment → Face Embedding (ArcFace) →")
    print("        Vector Matching (Cosine Similarity)")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Database: {database_path}")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print()
    
    # Initialize recognizer
    try:
        recognizer = OneShotRecognizer(
            database_path=database_path,
            similarity_threshold=similarity_threshold
        )
    except Exception as e:
        print(f"❌ Failed to initialize recognizer: {e}")
        return False
    
    # Check database
    stats = recognizer.get_database_stats()
    if stats['total_references'] == 0:
        print("❌ Reference database is empty!")
        print("   Build database first using: python build_reference_database.py")
        return False
    
    print(f"📊 Database: {stats['total_references']} references")
    print(f"   Names: {', '.join(stats['names'])}")
    print()
    
    # Recognize face
    print("🔍 Processing image...")
    print("   Step 1: Image Preprocessing (CLAHE)")
    print("   Step 2: Face Detection (RetinaFace)")
    print("   Step 3: Face Quality Filtering")
    print("   Step 4: Face Alignment")
    print("   Step 5: Face Embedding (ArcFace)")
    print("   Step 6: Vector Matching (Cosine Similarity)")
    print()
    
    results = recognizer.recognize_from_image(str(image_path), top_k=top_k)
    
    if results is None:
        print("❌ No face detected in the image")
        return False
    
    # Display results
    print("📊 RECOGNITION RESULTS:")
    print("-" * 70)
    
    best_match = results[0]
    status_icon = "✅" if best_match['is_match'] else "❌"
    print(f"{status_icon} Best Match: {best_match['name']}")
    print(f"   Similarity: {best_match['similarity']:.1%}")
    print(f"   Status: {'MATCH' if best_match['is_match'] else 'NO MATCH (below threshold)'}")
    
    if len(results) > 1:
        print(f"\n📋 Top {min(top_k, len(results))} Matches:")
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result['is_match'] else "❌"
            print(f"   {i}. {result['name']}: {result['similarity']:.1%} {status_icon}")
    
    # Display image if requested
    if show_image:
        img = cv2.imread(str(image_path))
        if img is not None:
            # Draw result on image
            label = f"{best_match['name']} ({best_match['similarity']:.1%})"
            color = (0, 255, 0) if best_match['is_match'] else (0, 0, 255)
            
            cv2.putText(img, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('One-Shot Face Recognition Result', img)
            print("\n📸 Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return True


def recognize_group_image(image_path: str,
                         database_path: str = "reference_database",
                         similarity_threshold: float = 0.6,
                         output_path: str = None,
                         show_image: bool = False) -> bool:
    """
    Recognize all faces in a group image.
    
    Args:
        image_path: Path to image file
        database_path: Path to reference database
        similarity_threshold: Minimum similarity for a match
        output_path: Path to save annotated image
        show_image: Whether to display the image
    
    Returns:
        True if successful, False otherwise
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print("=" * 70)
    print("👥 GROUP FACE RECOGNITION (ONE-SHOT LEARNING)")
    print("=" * 70)
    print("System: Image → Preprocessing (CLAHE) → Face Detection (RetinaFace) →")
    print("        Quality Filter → Face Alignment → Face Embedding (ArcFace) →")
    print("        Vector Matching (Cosine Similarity)")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Database: {database_path}")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print()
    
    # Initialize recognizer
    try:
        recognizer = OneShotRecognizer(
            database_path=database_path,
            similarity_threshold=similarity_threshold
        )
    except Exception as e:
        print(f"❌ Failed to initialize recognizer: {e}")
        return False
    
    # Check database
    stats = recognizer.get_database_stats()
    if stats['total_references'] == 0:
        print("❌ Reference database is empty!")
        print("   Build database first using: python build_reference_database.py")
        return False
    
    print(f"📊 Database: {stats['total_references']} references")
    print()
    
    # Recognize all faces
    print("🔍 Processing image...")
    print("   Step 1: Image Preprocessing (CLAHE)")
    print("   Step 2: Face Detection (RetinaFace)")
    print("   Step 3: Face Quality Filtering")
    print("   Step 4: Face Alignment")
    print("   Step 5: Face Embedding (ArcFace)")
    print("   Step 6: Vector Matching (Cosine Similarity)")
    print()
    
    faces = recognizer.recognize_multiple_faces(str(image_path))
    
    if not faces:
        print("❌ No faces detected in the image")
        return False
    
    print(f"✅ Detected {len(faces)} face(s)")
    print()
    
    # Load image for annotation
    img = cv2.imread(str(image_path))
    if img is None:
        print("❌ Failed to load image for annotation")
        return False
    
    # Draw results on image
    print("📊 RECOGNITION RESULTS:")
    print("-" * 70)
    
    for i, face_info in enumerate(faces, 1):
        recognition = face_info['recognition']
        bbox = face_info['bbox']
        x1, y1, x2, y2 = bbox
        
        if recognition:
            # Choose color based on match status
            if recognition['is_match']:
                color = (0, 255, 0)  # Green - match
            else:
                color = (0, 0, 255)  # Red - no match
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{recognition['name']} ({recognition['similarity']:.1%})"
            
            # Draw label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y2 + 25
            
            # Draw label background
            cv2.rectangle(img,
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0] + 10, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Print result
            status_icon = "✅ MATCH" if recognition['is_match'] else "❌ NO MATCH"
            print(f"👤 Face {i}: {recognition['name']} ({recognition['similarity']:.1%}) {status_icon}")
        else:
            # No recognition result
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 2)
            print(f"👤 Face {i}: Unknown (no recognition result)")
    
    # Save annotated image
    if output_path:
        output_path = Path(output_path)
    else:
        # Auto-save with _annotated suffix
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    
    cv2.imwrite(str(output_path), img)
    print(f"\n💾 Annotated image saved: {output_path}")
    
    # Display image if requested
    if show_image:
        cv2.imshow('Group Face Recognition Result', img)
        print("\n📸 Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Recognize faces using one-shot learning approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recognize single face
  python recognize_one_shot.py --image data/test/person_1.jpg
  
  # Recognize with custom threshold
  python recognize_one_shot.py --image data/test/person_1.jpg --threshold 0.7
  
  # Recognize group image
  python recognize_one_shot.py --image data/test/gtest1.jpg --group
  
  # Show image
  python recognize_one_shot.py --image data/test/person_1.jpg --show

System Architecture:
  Image → Preprocessing (CLAHE) → Face Detection (RetinaFace) → 
  Quality Filter → Face Alignment → Face Embedding (ArcFace) → 
  Vector Matching (Cosine Similarity)
        """
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--database', '-d', type=str, default='databases/reference_database',
                       help='Path to reference database (default: databases/reference_database)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                       help='Similarity threshold for matching (default: 0.6)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top matches to show (default: 3)')
    parser.add_argument('--group', '-g', action='store_true',
                       help='Process as group image (multiple faces)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for annotated image (group mode only)')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Display the result image')
    parser.add_argument('--model', '-m', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_s'],
                       help='InsightFace model name (default: buffalo_l)')
    parser.add_argument('--ctx_id', type=int, default=-1,
                       help='Device ID (-1 for CPU, 0+ for GPU)')
    
    args = parser.parse_args()
    
    if args.group:
        success = recognize_group_image(
            image_path=args.image,
            database_path=args.database,
            similarity_threshold=args.threshold,
            output_path=args.output,
            show_image=args.show
        )
    else:
        success = recognize_single_face(
            image_path=args.image,
            database_path=args.database,
            similarity_threshold=args.threshold,
            top_k=args.top_k,
            show_image=args.show
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

