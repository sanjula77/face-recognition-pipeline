"""
Similarity Computation Module
Functions for computing similarity between face embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0-1, where 1 is identical)
    """
    # Normalize embeddings
    emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # Clamp to [0, 1] range (cosine similarity can be negative)
    similarity = max(0.0, min(1.0, similarity))
    
    return float(similarity)


def compute_euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Euclidean distance (lower is more similar)
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    return float(distance)


def find_best_match(query_embedding: np.ndarray,
                   reference_embeddings: np.ndarray,
                   reference_names: List[str],
                   threshold: float = 0.6,
                   top_k: int = 1) -> List[Dict]:
    """
    Find best matching reference for a query embedding using cosine similarity.
    
    Args:
        query_embedding: Query face embedding (512-dimensional)
        reference_embeddings: Array of reference embeddings (n, 512)
        reference_names: List of names corresponding to reference embeddings
        threshold: Minimum similarity threshold for a match
        top_k: Number of top matches to return
    
    Returns:
        List of match results, each containing:
        - name: Person's name (or 'Unknown' if below threshold)
        - similarity: Similarity score (0-1)
        - index: Index in reference list
        - is_match: Whether similarity meets threshold
    """
    if len(reference_embeddings) == 0:
        return [{
            'name': 'Unknown',
            'similarity': 0.0,
            'index': -1,
            'is_match': False,
            'reason': 'No reference embeddings available'
        }]
    
    if len(reference_embeddings) != len(reference_names):
        raise ValueError("Number of embeddings must match number of names")
    
    # Normalize query embedding
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    # Normalize reference embeddings
    ref_norms = reference_embeddings / (np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarities
    similarities = np.dot(ref_norms, query_norm)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        similarity = float(similarities[idx])
        
        results.append({
            'name': reference_names[idx] if similarity >= threshold else 'Unknown',
            'similarity': similarity,
            'index': int(idx),
            'is_match': similarity >= threshold
        })
    
    return results


def batch_find_matches(query_embeddings: np.ndarray,
                      reference_embeddings: np.ndarray,
                      reference_names: List[str],
                      threshold: float = 0.6,
                      top_k: int = 1) -> List[List[Dict]]:
    """
    Find best matches for multiple query embeddings.
    
    Args:
        query_embeddings: Array of query embeddings (n, 512)
        reference_embeddings: Array of reference embeddings (m, 512)
        reference_names: List of names corresponding to reference embeddings
        threshold: Minimum similarity threshold
        top_k: Number of top matches per query
    
    Returns:
        List of match results for each query embedding
    """
    results = []
    for query_emb in query_embeddings:
        matches = find_best_match(
            query_emb,
            reference_embeddings,
            reference_names,
            threshold,
            top_k
        )
        results.append(matches)
    
    return results

