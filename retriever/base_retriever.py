from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BaseRetriever(ABC):
    """Abstract base class for all retrievers.

    Each retriever accepts a natural language query and returns top-k matching images
    from a pre-built FAISS index. Subclasses implement the specific search logic.
    """

    def __init__(self, version_name: str, index_dir: str, data_dir: str):
        self.version_name = version_name
        self.index_dir = index_dir
        self.data_dir = data_dir
        self.image_paths: List[str] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load image paths metadata from index directory."""
        import json
        from pathlib import Path
        meta_path = Path(self.index_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.image_paths = meta.get("image_paths", [])
        else:
            # Fallback: scan data directory
            extensions = {".jpg", ".jpeg", ".png", ".webp"}
            self.image_paths = sorted([
                str(p) for p in Path(self.data_dir).rglob("*")
                if p.suffix.lower() in extensions
            ])

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images for a natural language query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: image_id, image_path, score, (optional: channel_scores)
        """
        pass

    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get full image path from image_id."""
        for path in self.image_paths:
            from pathlib import Path
            if Path(path).stem == image_id:
                return path
        return None
