from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class BaseIndexer(ABC):
    """Abstract base class for all indexers.

    Each indexer processes images into vector representations and stores them
    in a FAISS index. Subclasses implement the specific feature extraction logic.
    """

    def __init__(self, version_name: str, embedding_dim: int, data_dir: str, output_dir: str):
        self.version_name = version_name
        self.embedding_dim = embedding_dim
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Extract feature vectors from images.

        Args:
            image_paths: List of paths to image files.
            batch_size: Batch size for model inference.

        Returns:
            Dict mapping image_id to numpy array of embeddings.
            For single-vector indexers: {image_id: np.ndarray of shape (dim,)}
            For multi-vector indexers: {image_id: dict of {vector_name: np.ndarray}}
        """
        pass

    @abstractmethod
    def build_index(self, features: Dict[str, Any]) -> None:
        """Build FAISS index from extracted features and save to disk."""
        pass

    def index(self, image_paths: List[str], batch_size: int = 32) -> None:
        """Full indexing pipeline: extract features → build index → save."""
        print(f"[{self.version_name}] Extracting features from {len(image_paths)} images...")
        features = self.extract_features(image_paths, batch_size)
        print(f"[{self.version_name}] Building FAISS index...")
        self.build_index(features)
        print(f"[{self.version_name}] Index saved to {self.output_dir}")

    def get_image_paths(self) -> List[str]:
        """Get all image paths from the data directory."""
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([
            str(p) for p in self.data_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])
        return paths

    @staticmethod
    def image_id_from_path(path: str) -> str:
        """Extract image_id from path (filename without extension)."""
        from pathlib import Path
        return Path(path).stem
