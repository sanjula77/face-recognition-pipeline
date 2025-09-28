# src/verification/test_normalization.py
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.detect_align import normalize_for_arcface

def test_normalization():
    """Test the normalization function with a sample image."""
    print("🧪 Testing ArcFace Normalization")
    print("=" * 50)
    
    # Create a sample 112x112 RGB image (uint8)
    sample_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    print(f"Input shape: {sample_img.shape}")
    print(f"Input dtype: {sample_img.dtype}")
    print(f"Input range: [{sample_img.min()}, {sample_img.max()}]")
    
    # Apply normalization
    normalized = normalize_for_arcface(sample_img)
    
    print(f"\nOutput shape: {normalized.shape}")
    print(f"Output dtype: {normalized.dtype}")
    print(f"Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Verify requirements
    print(f"\n✅ Verification:")
    print(f"   Shape is (3, 112, 112): {normalized.shape == (3, 112, 112)}")
    print(f"   Dtype is float32: {normalized.dtype == np.float32}")
    print(f"   Range is [-1, 1]: {normalized.min() >= -1.0 and normalized.max() <= 1.0}")
    print(f"   CHW format: {normalized.shape[0] == 3}")  # Channel first
    
    return normalized

def verify_npy_files(processed_dir="data/processed"):
    """Verify the .npy files created by the pipeline."""
    print(f"\n🔍 Verifying .npy files in {processed_dir}")
    print("=" * 50)
    
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        print(f"❌ Directory {processed_dir} not found!")
        return
    
    npy_files = list(processed_path.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files")
    
    for npy_file in npy_files:
        try:
            # Load the .npy file
            data = np.load(npy_file)
            
            print(f"\n📁 {npy_file.name}:")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"   Mean: {data.mean():.3f}")
            print(f"   Std: {data.std():.3f}")
            
            # Verify ArcFace requirements
            is_valid = (
                data.shape == (3, 112, 112) and
                data.dtype == np.float32 and
                data.min() >= -1.0 and
                data.max() <= 1.0
            )
            print(f"   ✅ ArcFace ready: {is_valid}")
            
        except Exception as e:
            print(f"   ❌ Error loading {npy_file.name}: {e}")

def load_for_arcface_example(npy_file_path):
    """Example of how to load normalized faces for ArcFace."""
    print(f"\n🚀 Loading {npy_file_path} for ArcFace:")
    
    # Load the normalized face
    face_tensor = np.load(npy_file_path)
    
    print(f"   Loaded tensor shape: {face_tensor.shape}")
    print(f"   Loaded tensor dtype: {face_tensor.dtype}")
    print(f"   Loaded tensor range: [{face_tensor.min():.3f}, {face_tensor.max():.3f}]")
    
    # This is ready for ArcFace embedding extraction
    print(f"   ✅ Ready for ArcFace embedding extraction!")
    
    return face_tensor

if __name__ == "__main__":
    # Test normalization function
    test_normalization()
    
    # Verify existing .npy files
    verify_npy_files()
    
    # Example usage
    processed_path = Path("data/processed")
    npy_files = list(processed_path.glob("*.npy"))
    if npy_files:
        load_for_arcface_example(npy_files[0])
