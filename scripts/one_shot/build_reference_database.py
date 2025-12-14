#!/usr/bin/env python3
"""
Build Reference Database
Extracts face embeddings from reference images (one per person) and builds a database.
System Architecture: Image → Face Detection (RetinaFace) → Face Alignment → Face Embedding (ArcFace) → Database
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
from src.one_shot_recognition.database import ReferenceDatabase
from src.one_shot_recognition.face_processor import FaceProcessor


def build_database(input_dir: str,
                  database_path: str = "databases/reference_database",
                  model_name: str = 'buffalo_l',
                  ctx_id: int = -1) -> bool:
    """
    Build reference database from images (one per person).
    
    Args:
        input_dir: Directory containing reference images (one per person)
        database_path: Path to store the database
        model_name: InsightFace model name
        ctx_id: Device ID (-1 for CPU, 0+ for GPU)
    
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Initialize components
    print("🔧 Initializing face processor...")
    face_processor = FaceProcessor(model_name=model_name, ctx_id=ctx_id)
    database = ReferenceDatabase(database_path)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix in image_extensions and f.is_file()
    ]
    
    if not image_files:
        print(f"❌ No image files found in: {input_dir}")
        return False
    
    print(f"\n📁 Found {len(image_files)} image(s) to process")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    
    for image_file in image_files:
        # Extract person name from filename (without extension)
        person_name = image_file.stem
        
        print(f"\n👤 Processing: {person_name}")
        print(f"   📷 Image: {image_file.name}")
        print(f"   🔧 Preprocessing: CLAHE → Detection → Quality Filter → Alignment → Embedding")
        
        try:
            # Process image: CLAHE → Detection → Quality Filter → Alignment → Embedding
            result = face_processor.process_image(str(image_file))
            
            if result is None:
                print(f"   ❌ No face detected in image")
                failed_count += 1
                continue
            
            # Add to database
            success = database.add_reference(
                name=person_name,
                embedding=result['embedding'],
                source_image=str(image_file),
                metadata={
                    'filename': image_file.name,
                    'file_path': str(image_file),
                    'detection_score': result['score'],
                    'bbox': result['bbox']
                }
            )
            
            if success:
                print(f"   ✅ Added reference for '{person_name}'")
                print(f"      Similarity threshold: 0.6 (adjustable)")
                success_count += 1
            else:
                print(f"   ❌ Failed to add reference")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            failed_count += 1
    
    # Save database
    print("\n" + "=" * 70)
    print(f"📊 Summary:")
    print(f"   ✅ Successfully processed: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    if success_count > 0:
        database.save()
        stats = database.get_statistics()
        print(f"\n✅ Reference database built successfully!")
        print(f"   Total references: {stats['total_references']}")
        print(f"   Database path: {stats['database_path']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
        print(f"   Names: {', '.join(stats['names'])}")
        print(f"\n💡 Next step: Use 'python recognize_one_shot.py --image <path>' to recognize faces")
        return True
    else:
        print("\n❌ No references were added to the database")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build reference database from images (one per person)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build database from reference images
  python build_reference_database.py --input_dir data/reference_images
  
  # Specify custom database path
  python build_reference_database.py --input_dir data/reference_images --database_path my_database
  
  # Use GPU
  python build_reference_database.py --input_dir data/reference_images --ctx_id 0
  
  # Use smaller model
  python build_reference_database.py --input_dir data/reference_images --model buffalo_s

System Architecture:
  Image → Face Detection (RetinaFace) → Face Alignment → Face Embedding (ArcFace) → Database
        """
    )
    
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Directory containing reference images (one per person)')
    parser.add_argument('--database_path', '-d', type=str, default='databases/reference_database',
                       help='Path to store the reference database (default: databases/reference_database)')
    parser.add_argument('--model', '-m', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_s'],
                       help='InsightFace model name (default: buffalo_l)')
    parser.add_argument('--ctx_id', type=int, default=-1,
                       help='Device ID (-1 for CPU, 0+ for GPU, default: -1)')
    
    args = parser.parse_args()
    
    print("🔧 BUILDING REFERENCE DATABASE")
    print("=" * 70)
    print("System Architecture:")
    print("  Image → Preprocessing (CLAHE) → Face Detection (RetinaFace) →")
    print("  Quality Filter → Face Alignment → Face Embedding (ArcFace) → Database")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Database path: {args.database_path}")
    print(f"Model: {args.model}")
    print(f"Device: {'GPU' if args.ctx_id >= 0 else 'CPU'}")
    print("=" * 70)
    
    success = build_database(
        input_dir=args.input_dir,
        database_path=args.database_path,
        model_name=args.model,
        ctx_id=args.ctx_id
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

