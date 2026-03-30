#!/usr/bin/env python3
"""PCA utilities for transforming query vectors at search time.

This module provides functions to load a pre-fitted PCA model and transform
new vectors to the reduced dimensionality space.
"""

import numpy as np
from pathlib import Path
from typing import Optional


class PCATransformer:
    """Lightweight PCA transformer for inference (no sklearn dependency at query time)."""
    
    def __init__(self, components: np.ndarray, mean: np.ndarray):
        """
        Args:
            components: PCA components matrix of shape (n_components, n_features)
            mean: Mean vector of shape (n_features,) used for centering
        """
        self.components = components.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.n_components = components.shape[0]
        self.n_features = components.shape[1]
    
    def transform(self, vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Transform vectors using the fitted PCA model.
        
        Args:
            vectors: Input vectors of shape (n_samples, n_features) or (n_features,)
            normalize: Whether to L2-normalize the output vectors
            
        Returns:
            Transformed vectors of shape (n_samples, n_components) or (n_components,)
        """
        single = vectors.ndim == 1
        if single:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self.n_features:
            raise ValueError(
                f"Input has {vectors.shape[1]} features, expected {self.n_features}"
            )
        
        # Center and project
        centered = vectors.astype(np.float32) - self.mean
        transformed = centered @ self.components.T
        
        if normalize:
            norms = np.linalg.norm(transformed, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            transformed = transformed / norms
        
        return transformed[0] if single else transformed.astype(np.float32)
    
    @classmethod
    def load(cls, path: str | Path) -> "PCATransformer":
        """Load a PCA transformer from a .npz file saved by extract_embeddings.py."""
        data = np.load(path)
        return cls(components=data["components"], mean=data["mean"])


def load_pca_model(model_path: str | Path) -> Optional[PCATransformer]:
    """Load PCA model from file, returning None if path doesn't exist."""
    path = Path(model_path)
    if not path.exists():
        return None
    return PCATransformer.load(path)


if __name__ == "__main__":
    # Simple test/demo
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PCA transformer")
    parser.add_argument("model_path", help="Path to .npz PCA model file")
    parser.add_argument("--test-dim", type=int, default=128,
                        help="Input dimension to test with random vectors")
    args = parser.parse_args()
    
    transformer = PCATransformer.load(args.model_path)
    print(f"Loaded PCA model: {args.test_dim}D -> {transformer.n_components}D")
    
    # Test with random vectors
    test_vectors = np.random.randn(5, args.test_dim).astype(np.float32)
    test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
    
    transformed = transformer.transform(test_vectors)
    print(f"Input shape: {test_vectors.shape}")
    print(f"Output shape: {transformed.shape}")
    print(f"Output norms (should be ~1): {np.linalg.norm(transformed, axis=1)}")
