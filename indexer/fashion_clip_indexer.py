"""V0b: Fashion-specific CLIP baseline indexer.

Uses Marqo-FashionSigLIP (ViT-B-16) — SOTA fashion CLIP fine-tuned with GCL.
Still single-vector, but domain-adapted. Shows what fine-tuning alone achieves
without architectural changes.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from indexer.base_indexer import BaseIndexer
from models.clip_models import MarqoFashionSigLIPWrapper
from utils.vector_store import VectorStore
from utils.image_utils import image_id_from_path


class FashionCLIPIndexer(BaseIndexer):
    """V0b: Marqo-FashionSigLIP single-vector indexer."""

    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        super().__init__(version_name="v0b", embedding_dim=768, data_dir=data_dir, output_dir=output_dir)
        self.clip = MarqoFashionSigLIPWrapper(device=device)

    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Extract single global Fashion-SigLIP embedding per image."""
        embeddings = self.clip.encode_images(image_paths, batch_size)
        features = {}
        for path, emb in zip(image_paths, embeddings):
            img_id = image_id_from_path(path)
            features[img_id] = emb
        return features

    def build_index(self, features: Dict[str, np.ndarray]) -> None:
        """Build FAISS HNSW index from Fashion-SigLIP embeddings."""
        image_ids = sorted(features.keys())
        embeddings = np.stack([features[img_id] for img_id in image_ids])

        store = VectorStore(dim=self.embedding_dim, metric="cosine")
        store.build(embeddings, image_ids)
        store.save(str(self.output_dir / "global"))

        meta = {"image_paths": getattr(self, '_image_paths', []),
                "version": self.version_name, "model": "Marqo-FashionSigLIP"}
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
