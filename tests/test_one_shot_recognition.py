#!/usr/bin/env python3
"""
Test One-Shot Face Recognition
Tests the one-shot recognition system on images from data/test/oneshortTest/
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

import argparse
from src.one_shot_recognition.recognizer import OneShotRecognizer


def extract_expected_name(filename):
    """Extract expected person name from filename"""
    # Remove extension
    name = Path(filename).stem
    
    # Remove numbers at the end (e.g., "bhanu1" -> "bhanu")
    while name and name[-1].isdigit():
        name = name[:-1]
    
    return name.lower()


def test_one_shot_recognition(test_dir: str = "data/test/oneshortTest",
                             database_path: str = "databases/reference_database",
                             similarity_threshold: float = 0.6,
                             show_details: bool = False):
    """Test one-shot recognition on test images"""
    
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
    print("🧪 ONE-SHOT FACE RECOGNITION TEST")
    print("=" * 70)
    print(f"Test directory: {test_dir}")
    print(f"Database: {database_path}")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print(f"Found {len(test_images)} test image(s)")
    print()
    
    # Initialize recognizer
    print("📦 Loading recognizer and database...")
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
        print("   Build database first using: python scripts/one_shot/build_reference_database.py")
        return False
    
    print(f"✅ Database loaded: {stats['total_references']} references")
    print(f"   Names: {', '.join(stats['names'])}")
    print()
    
    # Test each image
    print("🔍 Testing images...")
    print("=" * 70)
    
    results = []
    correct_predictions = 0
    total_similarity = 0.0
    no_face_count = 0
    unknown_count = 0
    
    for img_path in sorted(test_images):
        # Extract expected name from filename
        expected_name = extract_expected_name(img_path.name)
        
        # Recognize face
        recognition_results = recognizer.recognize_from_image(str(img_path), top_k=1)
        
        if recognition_results is None:
            print(f"❌ {img_path.name:30} → No face detected")
            results.append({
                'image': img_path.name,
                'expected': expected_name,
                'predicted': 'No face',
                'similarity': 0.0,
                'correct': False
            })
            no_face_count += 1
            continue
        
        best_match = recognition_results[0]
        predicted_name = best_match['name']
        similarity = best_match['similarity']
        is_match = best_match['is_match']
        
        # Check if correct
        is_correct = False
        if is_match and predicted_name.lower() != 'unknown':
            is_correct = (predicted_name.lower() == expected_name.lower())
            if is_correct:
                correct_predictions += 1
                total_similarity += similarity
        
        if predicted_name == 'Unknown':
            unknown_count += 1
        
        # Store result
        results.append({
            'image': img_path.name,
            'expected': expected_name,
            'predicted': predicted_name,
            'similarity': similarity,
            'is_match': is_match,
            'correct': is_correct
        })
        
        # Print result
        if is_correct:
            status = "✅"
        elif not is_match or predicted_name == 'Unknown':
            status = "⚠️"
        else:
            status = "❌"
        
        match_status = "MATCH" if is_match else "NO MATCH"
        print(f"{status} {img_path.name:30} → {predicted_name:15} ({similarity:5.1%}) [{match_status}] [Expected: {expected_name}]")
        
        if show_details and not is_correct and recognition_results:
            # Show top 3 matches
            if len(recognition_results) > 1:
                print(f"   Top matches: ", end="")
                for i, result in enumerate(recognition_results[:3], 1):
                    print(f"{i}. {result['name']} ({result['similarity']:.1%})", end="  ")
                print()
    
    # Calculate statistics
    total_images = len(results)
    faces_detected = total_images - no_face_count
    accuracy = correct_predictions / faces_detected if faces_detected > 0 else 0.0
    avg_similarity = total_similarity / correct_predictions if correct_predictions > 0 else 0.0
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total images tested: {total_images}")
    print(f"Faces detected: {faces_detected}")
    print(f"No face detected: {no_face_count}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {faces_detected - correct_predictions}")
    print(f"Unknown (below threshold): {unknown_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average similarity (correct): {avg_similarity:.1%}")
    print()
    
    # Show incorrect predictions
    incorrect = [r for r in results if not r['correct'] and r['predicted'] != 'No face']
    if incorrect:
        print("❌ Incorrect Predictions:")
        for r in incorrect:
            match_status = "MATCH" if r['is_match'] else "NO MATCH"
            print(f"   {r['image']:30} → Predicted: {r['predicted']:15} (Expected: {r['expected']}, Similarity: {r['similarity']:.1%}, {match_status})")
        print()
    
    # Show no face cases
    if no_face_count > 0:
        print("⚠️  Images with no face detected:")
        for r in results:
            if r['predicted'] == 'No face':
                print(f"   {r['image']}")
        print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test one-shot face recognition on test images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  python tests/test_one_shot_recognition.py
  
  # Test with custom directory and threshold
  python tests/test_one_shot_recognition.py --test_dir data/test/oneshortTest --threshold 0.7
  
  # Show detailed results
  python tests/test_one_shot_recognition.py --details
        """
    )
    
    parser.add_argument('--test_dir', '-t', type=str, default='data/test/oneshortTest',
                       help='Directory containing test images (default: data/test/oneshortTest)')
    parser.add_argument('--database', '-d', type=str, default='databases/reference_database',
                       help='Path to reference database (default: databases/reference_database)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold for matching (default: 0.6)')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed results for incorrect predictions')
    
    args = parser.parse_args()
    
    success = test_one_shot_recognition(
        test_dir=args.test_dir,
        database_path=args.database,
        similarity_threshold=args.threshold,
        show_details=args.details
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

