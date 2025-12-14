"""
Enhanced Embedding Normalization Module
Multiple normalization strategies for better embedding quality
"""

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from typing import Optional, Literal, List

class EmbeddingNormalizer:
    """
    Enhanced embedding normalization with multiple strategies.
    """
    
    def __init__(self, method: Literal['l2', 'l1', 'zscore', 'minmax', 'robust', 'combined'] = 'l2'):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method
                - 'l2': L2 normalization (unit length)
                - 'l1': L1 normalization
                - 'zscore': Z-score normalization (mean=0, std=1)
                - 'minmax': Min-max normalization to [0, 1]
                - 'robust': Robust normalization (median, IQR)
                - 'combined': L2 + Z-score combination
        """
        self.method = method
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit normalizer on training data (for methods that need it).
        
        Args:
            X: Training embeddings
        """
        if self.method in ['zscore', 'minmax', 'robust', 'combined']:
            if self.method == 'zscore' or self.method == 'combined':
                self.scaler = StandardScaler()
            elif self.method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            
            self.scaler.fit(X)
            self.is_fitted = True
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings.
        
        Args:
            X: Embeddings to normalize
        
        Returns:
            Normalized embeddings
        """
        if self.method == 'l2':
            return normalize(X, norm='l2', axis=1)
        
        elif self.method == 'l1':
            return normalize(X, norm='l1', axis=1)
        
        elif self.method == 'zscore':
            if not self.is_fitted:
                # Fit on the data if not already fitted
                self.fit(X)
            return self.scaler.transform(X)
        
        elif self.method == 'minmax':
            if not self.is_fitted:
                self.fit(X)
            return self.scaler.transform(X)
        
        elif self.method == 'robust':
            if not self.is_fitted:
                self.fit(X)
            return self.scaler.transform(X)
        
        elif self.method == 'combined':
            # L2 normalize first, then z-score
            X_l2 = normalize(X, norm='l2', axis=1)
            if not self.is_fitted:
                self.fit(X_l2)
            return self.scaler.transform(X_l2)
        
        else:
            # Default to L2
            return normalize(X, norm='l2', axis=1)
    
    def normalize_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize a single embedding vector.
        
        Args:
            embedding: Single embedding vector
        
        Returns:
            Normalized embedding
        """
        return self.normalize(embedding.reshape(1, -1))[0]

def get_best_normalization_method(X_train: np.ndarray, y_train: np.ndarray, 
                                  test_methods: Optional[List] = None) -> str:
    """
    Find the best normalization method using cross-validation.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        test_methods: List of methods to test (default: all)
    
    Returns:
        Best normalization method
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    if test_methods is None:
        test_methods = ['l2', 'zscore', 'combined']
    
    best_method = 'l2'
    best_score = 0.0
    
    for method in test_methods:
        normalizer = EmbeddingNormalizer(method=method)
        X_norm = normalizer.normalize(X_train)
        
        # Quick test with simple classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_norm, y_train, cv=3, scoring='accuracy')
        avg_score = scores.mean()
        
        print(f"   {method:10} normalization: {avg_score:.4f} accuracy")
        
        if avg_score > best_score:
            best_score = avg_score
            best_method = method
    
    print(f"✅ Best normalization method: {best_method} (accuracy: {best_score:.4f})")
    return best_method

