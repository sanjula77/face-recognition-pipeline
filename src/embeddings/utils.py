# src/embeddings/utils.py
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mlflow

def save_embedding(embedding: np.ndarray, metadata: Dict, output_path: str):
    """
    Save face embedding and metadata.
    
    Args:
        embedding: Face embedding vector (512-dimensional for ArcFace)
        metadata: Metadata dictionary
        output_path: Path to save the embedding (.npy file)
    """
    # Save embedding as .npy file
    np.save(output_path, embedding)
    
    # Save metadata as .json file
    meta_path = output_path.replace('.npy', '.meta.json')
    with open(meta_path, 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_embedding(embedding_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load face embedding and metadata.
    
    Args:
        embedding_path: Path to the .npy embedding file
        
    Returns:
        Tuple of (embedding, metadata)
    """
    # Load embedding
    embedding = np.load(embedding_path)
    
    # Load metadata
    meta_path = str(embedding_path).replace('.npy', '.meta.json')
    with open(meta_path, 'r', encoding='utf8') as f:
        metadata = json.load(f)
    
    return embedding, metadata

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector to unit length.
    
    Args:
        embedding: Raw embedding vector
        
    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize embeddings
    norm1 = normalize_embedding(embedding1)
    norm2 = normalize_embedding(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(norm1, norm2)
    return float(similarity)

def batch_extract_embeddings(processed_dir: str, output_dir: str, model) -> Dict:
    """
    Extract embeddings for all processed face images.
    
    Args:
        processed_dir: Directory containing processed .npy face files
        output_dir: Directory to save embeddings
        model: ArcFace model for embedding extraction
        
    Returns:
        Dictionary with extraction statistics
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .npy face files in subdirectories
    face_files = list(processed_path.rglob("*.npy"))
    
    stats = {
        'total_faces': len(face_files),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'embeddings_saved': []
    }
    
    print(f"🔍 Found {len(face_files)} face files to process")
    
    for face_file in face_files:
        try:
            # Load normalized face
            face_tensor = np.load(face_file)
            
            # Extract embedding using ArcFace recognition model directly
            # Convert face tensor to the format expected by insightface
            face_img = np.transpose(face_tensor, (1, 2, 0))  # CHW to HWC
            face_img = ((face_img + 1) * 127.5).astype(np.uint8)  # [-1,1] to [0,255]
            
            # Use recognition model directly for pre-aligned faces
            embedding = model.models['recognition'].get_feat(face_img)
            # Flatten embedding to 1D vector
            embedding = embedding.flatten()
            
            # Load original metadata
            meta_file = face_file.with_suffix('.npy').with_suffix('.meta.json')
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Add embedding metadata
            metadata.update({
                'embedding_shape': embedding.shape,
                'embedding_dtype': str(embedding.dtype),
                'embedding_norm': float(np.linalg.norm(embedding)),
                'source_face_file': str(face_file)
            })
            
            # Save embedding with person name to avoid conflicts
            # Extract person name from the path (e.g., data/processed/ameesha/1_face0.npy -> ameesha)
            person_name = face_file.parent.name
            embedding_file = output_path / f"{person_name}_{face_file.stem}.embedding.npy"
            save_embedding(embedding, metadata, str(embedding_file))
            
            stats['successful_extractions'] += 1
            stats['embeddings_saved'].append(str(embedding_file))
            
            print(f"✅ Extracted embedding: {face_file.name}")
            
        except Exception as e:
            print(f"❌ Failed to extract embedding from {face_file.name}: {e}")
            stats['failed_extractions'] += 1
    
    return stats

def create_embedding_database(embeddings_dir: str, output_file: str):
    """
    Create a database of all embeddings for similarity search.
    
    Args:
        embeddings_dir: Directory containing embedding files
        output_file: Path to save the database
    """
    embeddings_path = Path(embeddings_dir)
    embedding_files = list(embeddings_path.glob("*.embedding.npy"))
    
    if not embedding_files:
        print("❌ No embedding files found!")
        return
    
    print(f"📊 Creating embedding database from {len(embedding_files)} files")
    
    # Load all embeddings
    embeddings = []
    metadata_list = []
    
    for embedding_file in embedding_files:
        embedding, metadata = load_embedding(embedding_file)
        embeddings.append(embedding)
        metadata_list.append(metadata)
    
    # Stack embeddings into a matrix
    embedding_matrix = np.vstack(embeddings)
    
    # Create database
    database = {
        'embeddings': embedding_matrix,
        'metadata': metadata_list,
        'file_paths': [str(f) for f in embedding_files],
        'total_embeddings': len(embeddings),
        'embedding_dim': embedding_matrix.shape[1]
    }
    
    # Save database
    np.savez_compressed(output_file, **database)
    
    print(f"✅ Database saved: {output_file}")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Embedding dimension: {embedding_matrix.shape[1]}")
    
    return database

def load_embeddings_database(embeddings_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load embeddings database and return X, y, labels for training.
    
    Args:
        embeddings_dir: Directory containing embedding files
        
    Returns:
        Tuple of (X, y, labels) where:
        - X: embedding matrix (n_samples, n_features)
        - y: label indices (n_samples,)
        - labels: list of unique person names
    """
    embeddings_path = Path(embeddings_dir)
    embedding_files = list(embeddings_path.glob("*.embedding.npy"))
    
    if not embedding_files:
        raise ValueError("No embedding files found!")
    
    print(f"📊 Loading {len(embedding_files)} embeddings")
    
    # Load all embeddings and extract labels
    embeddings = []
    person_names = []
    
    for embedding_file in embedding_files:
        embedding, metadata = load_embedding(embedding_file)
        embeddings.append(embedding)
        
        # Extract person name from filename
        filename = embedding_file.stem.replace('.embedding', '')
        if '_face' in filename:
            # Extract just the first part (person name) before any other underscores
            full_prefix = filename.split('_face')[0]
            person_name = full_prefix.split('_')[0]
        else:
            person_name = filename.split('_')[0]
        person_names.append(person_name)
    
    # Convert to arrays
    X = np.vstack(embeddings)
    
    # Create label mapping
    unique_labels = sorted(list(set(person_names)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[name] for name in person_names])
    
    print(f"✅ Loaded {X.shape[0]} embeddings for {len(unique_labels)} people")
    print(f"   People: {unique_labels}")
    
    return X, y, unique_labels