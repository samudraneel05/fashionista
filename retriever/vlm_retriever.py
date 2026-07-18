"""V3: VLM captioning + dual-path retriever.

Stage 1: BGE text similarity (query text vs caption text) → top-50
Stage 2: CLIP cross-modal re-ranking (query text vs candidate images) → top-k

The text-to-text stage naturally handles compositionality and context
since VLM captions explicitly state "red shirt AND blue pants in an office".
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from retriever.base_retriever import BaseRetriever
from models.clip_models import CLIPWrapper
from utils.vector_store import VectorStore


class VLMRetriever(BaseRetriever):
    """V3: VLM captioning retriever with two-stage text + cross-modal search."""

    def __init__(self, index_dir: str, data_dir: str, device: str = None):
        super().__init__(version_name="v3", index_dir=index_dir, data_dir=data_dir)
        self.device = device

        # BGE text encoder for query encoding
        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

        # CLIP for cross-modal re-ranking
        self.clip = CLIPWrapper(device=device)

        # Load caption index
        self.store = VectorStore(dim=1024, metric="cosine")
        self.store.load(str(Path(index_dir) / "caption"))

        # Load captions
        import json
        captions_path = Path(index_dir) / "captions.json"
        self.captions = {}
        if captions_path.exists():
            with open(captions_path, "r") as f:
                self.captions = json.load(f)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images using two-stage text + cross-modal search."""
        # Stage 1: BGE text similarity → top-50
        query_vec = self.text_encoder.encode(
            [query], normalize_embeddings=True
        )[0].astype(np.float32)

        recall_k = max(top_k * 5, 50)
        text_results = self.store.search(query_vec, top_k=recall_k)

        if not text_results:
            return []

        # Stage 2: CLIP cross-modal re-ranking
        candidate_ids = [img_id for img_id, _ in text_results]
        text_scores = {img_id: score for img_id, score in text_results}

        # Encode query with CLIP text encoder
        clip_query_vec = self.clip.encode_text(query)

        # Encode candidate images with CLIP vision encoder and compute similarity
        clip_scores = {}
        for img_id in candidate_ids:
            img_path = self.get_image_path(img_id)
            if img_path:
                try:
                    img_vec = self.clip.encode_image(img_path)
                    # Cosine similarity (both are L2-normalized)
                    sim = float(np.dot(clip_query_vec, img_vec))
                    clip_scores[img_id] = sim
                except Exception:
                    clip_scores[img_id] = 0.0
            else:
                clip_scores[img_id] = 0.0

        # Combine scores: weighted fusion of text similarity and cross-modal similarity
        text_weight = 0.6
        clip_weight = 0.4

        scored = []
        for img_id in candidate_ids:
            s_text = text_scores.get(img_id, 0.0)
            s_clip = clip_scores.get(img_id, 0.0)
            final_score = text_weight * s_text + clip_weight * s_clip

            scored.append({
                "image_id": img_id,
                "image_path": self.get_image_path(img_id),
                "score": final_score,
                "channel_scores": {
                    "text_similarity": s_text,
                    "clip_cross_modal": s_clip,
                },
                "caption": self.captions.get(img_id, {}).get("combined", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
