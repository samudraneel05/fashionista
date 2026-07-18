"""FAISS vector store wrapper for efficient similarity search."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss


class VectorStore:
    """Wrapper around FAISS for building, saving, loading, and searching vector indexes.

    Uses HNSW (Hierarchical Navigable Small World) index for sub-linear search
    that scales to millions of vectors.
    """

    def __init__(self, dim: int, metric: str = "cosine"):
        """Initialize vector store.

        Args:
            dim: Embedding dimensionality.
            metric: Similarity metric — 'cosine' or 'l2'.
        """
        self.dim = dim
        self.metric = metric
        self.index: Optional[faiss.Index] = None
        self.image_ids: List[str] = []

    def build(self, embeddings: np.ndarray, image_ids: List[str], m: int = 16, ef_construction: int = 100) -> None:
        """Build HNSW index from embeddings.

        Args:
            embeddings: numpy array of shape (N, dim), L2-normalized if cosine.
            image_ids: List of image IDs corresponding to each row.
            m: HNSW graph degree (higher = more accurate, more memory).
            ef_construction: Search depth during construction (higher = better quality).
        """
        assert embeddings.shape[1] == self.dim, f"Expected dim {self.dim}, got {embeddings.shape[1]}"
        assert len(image_ids) == embeddings.shape[0], "Mismatch between embeddings and image_ids"

        if self.metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_L2)

        self.index.hnsw.efConstruction = ef_construction
        self.index.add(embeddings.astype(np.float32))
        self.image_ids = image_ids

    def search(self, query_vec: np.ndarray, top_k: int = 10, ef_search: int = 100) -> List[Tuple[str, float]]:
        """Search for top-k similar vectors.

        Args:
            query_vec: Query embedding of shape (dim,) or (1, dim).
            top_k: Number of results.
            ef_search: Search depth (higher = more accurate, slower).

        Returns:
            List of (image_id, score) tuples, sorted by descending score.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(query_vec)

        self.index.hnsw.efSearch = ef_search
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.image_ids):
                results.append((self.image_ids[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "image_ids.json", "w") as f:
            json.dump(self.image_ids, f)

        meta = {"dim": self.dim, "metric": self.metric, "num_vectors": len(self.image_ids)}
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "image_ids.json", "r") as f:
            self.image_ids = json.load(f)

        with open(path / "meta.json", "r") as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.metric = meta["metric"]

    @staticmethod
    def load_multi(path: str) -> Dict[str, "VectorStore"]:
        """Load multiple named vector stores from a directory.

        Expected structure:
            path/
              global/
                index.faiss, image_ids.json, meta.json
              scene/
                index.faiss, image_ids.json, meta.json
              ...
        """
        path = Path(path)
        stores = {}
        for child in path.iterdir():
            if child.is_dir() and (child / "index.faiss").exists():
                vs = VectorStore(dim=0)
                vs.load(str(child))
                stores[child.name] = vs
        return stores


class MultiVectorStore:
    """Manages multiple FAISS indexes for multi-vector retrieval.

    Stores multiple named vector stores (e.g., 'global', 'scene', 'action')
    and provides unified search across all channels.
    """

    def __init__(self, dim: int, channel_names: List[str], metric: str = "cosine"):
        self.dim = dim
        self.channel_names = channel_names
        self.metric = metric
        self.stores: Dict[str, VectorStore] = {
            name: VectorStore(dim, metric) for name in channel_names
        }
        self.image_ids: List[str] = []

    def build(self, features: Dict[str, Dict[str, np.ndarray]], image_ids: List[str]) -> None:
        """Build all channel indexes.

        Args:
            features: {image_id: {channel_name: np.ndarray of shape (dim,)}}
            image_ids: List of image IDs in order.
        """
        self.image_ids = image_ids

        for channel in self.channel_names:
            embeddings = np.stack([features[img_id][channel] for img_id in image_ids])
            self.stores[channel].build(embeddings, image_ids)

    def search_channel(self, channel: str, query_vec: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """Search a single channel."""
        return self.stores[channel].search(query_vec, top_k)

    def search_all_channels(self, query_vecs: Dict[str, np.ndarray], top_k: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        """Search all channels with their respective query vectors.

        Args:
            query_vecs: {channel_name: query_embedding}
            top_k: Number of results per channel.

        Returns:
            {channel_name: [(image_id, score), ...]}
        """
        results = {}
        for channel, qvec in query_vecs.items():
            if channel in self.stores:
                results[channel] = self.stores[channel].search(qvec, top_k)
        return results

    def save(self, path: str) -> None:
        """Save all channel indexes to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for channel, store in self.stores.items():
            store.save(str(path / channel))

        meta = {
            "dim": self.dim,
            "channel_names": self.channel_names,
            "metric": self.metric,
            "image_ids": self.image_ids,
        }
        with open(path / "multivector_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load all channel indexes from disk."""
        path = Path(path)

        with open(path / "multivector_meta.json", "r") as f:
            meta = json.load(f)

        self.dim = meta["dim"]
        self.channel_names = meta["channel_names"]
        self.metric = meta["metric"]
        self.image_ids = meta["image_ids"]
        self.stores = {}

        for channel in self.channel_names:
            vs = VectorStore(dim=self.dim, metric=self.metric)
            vs.load(str(path / channel))
            self.stores[channel] = vs
