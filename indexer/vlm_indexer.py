"""V3: VLM captioning + dual-path retrieval indexer.

Uses Florence-2 for structured captions and BLIP for natural language captions.
Combines both into a rich text representation, then embeds with BGE-large.

Captions naturally encode compositionality ("red shirt AND blue pants") and
context ("in a modern office") in text space.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from indexer.base_indexer import BaseIndexer
from models.vlm_captioner import VLMCaptioner
from utils.vector_store import VectorStore
from utils.image_utils import image_id_from_path, load_image


class VLMIndexer(BaseIndexer):
    """V3: VLM captioning indexer with BGE text embeddings."""

    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        super().__init__(version_name="v3", embedding_dim=1024, data_dir=data_dir, output_dir=output_dir)
        self.vlm = VLMCaptioner(device=device)

        # BGE-large text encoder
        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

        # Captions storage
        self.captions: Dict[str, Dict[str, str]] = {}

    def extract_features(self, image_paths: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """Generate VLM captions and embed them with BGE.

        Returns:
            {image_id: np.ndarray(1024,)} — BGE text embeddings of combined captions.
        """
        captions = {}

        for path in tqdm(image_paths, desc=f"[{self.version_name}] Captioning images"):
            img_id = image_id_from_path(path)
            image = np.array(load_image(path))

            try:
                caps = self.vlm.caption(image)
                captions[img_id] = caps
            except Exception as e:
                print(f"Error captioning {path}: {e}")
                captions[img_id] = {"florence": "", "blip": "", "combined": ""}

        self.captions = captions

        # Embed combined captions with BGE
        image_ids = sorted(captions.keys())
        texts = [captions[img_id]["combined"] for img_id in image_ids]

        print(f"[{self.version_name}] Encoding {len(texts)} captions with BGE-large...")
        embeddings = self.text_encoder.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=True,
        )

        features = {}
        for img_id, emb in zip(image_ids, embeddings):
            features[img_id] = emb.astype(np.float32)

        return features

    def build_index(self, features: Dict[str, np.ndarray]) -> None:
        """Build FAISS index from BGE text embeddings and save captions."""
        image_ids = sorted(features.keys())
        embeddings = np.stack([features[img_id] for img_id in image_ids])

        store = VectorStore(dim=self.embedding_dim, metric="cosine")
        store.build(embeddings, image_ids)
        store.save(str(self.output_dir / "caption"))

        # Save captions for retrieval and evaluation
        captions_path = self.output_dir / "captions.json"
        with open(captions_path, "w") as f:
            json.dump(self.captions, f, indent=2)

        # Save metadata
        image_paths = getattr(self, '_image_paths', [])
        meta = {
            "image_paths": image_paths,
            "version": self.version_name,
            "model": "Florence-2 + BLIP + BGE-large-en-v1.5",
            "embedding_dim": self.embedding_dim,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
