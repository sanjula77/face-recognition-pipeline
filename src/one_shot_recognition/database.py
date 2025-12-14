"""
Reference Database Module
Manages storage and retrieval of reference face embeddings for one-shot recognition.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ReferenceDatabase:
    """
    Database for storing reference face embeddings.
    Each entry represents one person with one reference image.
    """
    
    def __init__(self, database_path: str = "databases/reference_database"):
        """
        Initialize reference database.
        
        Args:
            database_path: Path to store the database files
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.database_path / "embeddings.npy"
        self.metadata_file = self.database_path / "metadata.json"
        
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.name_to_index: Dict[str, int] = {}
        
        # Load existing database if available
        self._load_database()
    
    def _load_database(self) -> bool:
        """
        Load existing database from disk.
        
        Returns:
            True if database was loaded, False otherwise
        """
        if self.embeddings_file.exists() and self.metadata_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # Build name to index mapping
                self.name_to_index = {
                    item['name']: idx for idx, item in enumerate(self.metadata)
                }
                
                print(f"✅ Loaded reference database: {len(self.metadata)} references")
                return True
            except Exception as e:
                print(f"⚠️  Failed to load database: {e}")
                return False
        return False
    
    def add_reference(self, name: str, embedding: np.ndarray,
                     source_image: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Add a new reference to the database.
        
        Args:
            name: Person's name
            embedding: Face embedding vector (512-dimensional)
            source_image: Path to source image (optional)
            metadata: Additional metadata (optional)
        
        Returns:
            True if added successfully, False otherwise
        """
        if embedding.ndim != 1 or len(embedding) != 512:
            raise ValueError(f"Embedding must be 1D array of length 512, got shape {embedding.shape}")
        
        # Normalize embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        if name in self.name_to_index:
            # Update existing reference
            idx = self.name_to_index[name]
            self.embeddings[idx] = embedding_norm
            self.metadata[idx].update({
                'name': name,
                'source_image': source_image or self.metadata[idx].get('source_image'),
                'updated_at': datetime.now().isoformat(),
                **(metadata or {})
            })
            print(f"✅ Updated reference for '{name}'")
        else:
            # Add new reference
            if self.embeddings is None:
                self.embeddings = embedding_norm.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding_norm])
            
            # Add metadata
            reference_meta = {
                'name': name,
                'source_image': source_image,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                **(metadata or {})
            }
            self.metadata.append(reference_meta)
            
            # Update mapping
            self.name_to_index[name] = len(self.metadata) - 1
            print(f"✅ Added reference for '{name}'")
        
        return True
    
    def remove_reference(self, name: str) -> bool:
        """
        Remove a reference from the database.
        
        Args:
            name: Person's name to remove
        
        Returns:
            True if removed successfully, False otherwise
        """
        if name not in self.name_to_index:
            print(f"⚠️  Reference for '{name}' not found")
            return False
        
        idx = self.name_to_index[name]
        
        # Remove from arrays
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        del self.metadata[idx]
        
        # Rebuild name to index mapping
        self.name_to_index = {
            item['name']: i for i, item in enumerate(self.metadata)
        }
        
        print(f"✅ Removed reference for '{name}'")
        return True
    
    def get_reference(self, name: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get reference by name.
        
        Args:
            name: Person's name
        
        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        if name not in self.name_to_index:
            return None
        
        idx = self.name_to_index[name]
        return self.embeddings[idx], self.metadata[idx]
    
    def get_all_references(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get all references.
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if self.embeddings is None:
            return np.array([]), []
        return self.embeddings, self.metadata
    
    def get_names(self) -> List[str]:
        """
        Get list of all person names in database.
        
        Returns:
            List of names
        """
        return list(self.name_to_index.keys())
    
    def save(self) -> bool:
        """
        Save database to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if self.embeddings is not None and len(self.embeddings) > 0:
                np.save(self.embeddings_file, self.embeddings)
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved reference database: {len(self.metadata)} references")
                return True
            else:
                print("⚠️  No references to save")
                return False
        except Exception as e:
            print(f"❌ Failed to save database: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all references from database.
        
        Returns:
            True if cleared successfully
        """
        self.embeddings = None
        self.metadata = []
        self.name_to_index = {}
        
        # Delete files
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        print("✅ Database cleared")
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_references': len(self.metadata),
            'names': list(self.name_to_index.keys()),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'database_path': str(self.database_path)
        }

