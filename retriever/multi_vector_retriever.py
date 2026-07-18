"""V1: Multi-vector semantic decomposition retriever.

Two-stage retrieval:
1. HNSW recall on v_global → top-50 candidates
2. Dynamic weighted scoring across global, scene, action vectors

Weights adapt based on query type (anatomical, contextual, style).
Inspired by FashionGlance (previous intern, selected).
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from retriever.base_retriever import BaseRetriever
from models.clip_models import MarqoFashionSigLIPWrapper
from utils.vector_store import MultiVectorStore
from utils.query_parser import QueryParser


class MultiVectorRetriever(BaseRetriever):
    """V1: Multi-vector retriever with dynamic weighted scoring."""

    WEIGHTS = {
        "anatomical": {"global": 0.20, "scene": 0.10, "action": 0.70},
        "contextual": {"global": 0.30, "scene": 0.50, "action": 0.20},
        "style": {"global": 0.60, "scene": 0.20, "action": 0.20},
        "compositional": {"global": 0.20, "scene": 0.10, "action": 0.70},
        "complex": {"global": 0.40, "scene": 0.30, "action": 0.30},
        "attribute": {"global": 0.20, "scene": 0.10, "action": 0.70},
    }

    SCENE_THRESHOLD = 0.1

    def __init__(self, index_dir: str, data_dir: str, device: str = None):
        super().__init__(version_name="v1", index_dir=index_dir, data_dir=data_dir)
        self.encoder = MarqoFashionSigLIPWrapper(device=device)
        self.store = MultiVectorStore(dim=512, channel_names=["global", "scene", "action"], metric="cosine")
        self.store.load(index_dir)
        self.parser = QueryParser()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images using two-stage multi-vector search."""
        # Parse query
        parsed = self.parser.parse(query)

        # Get weights based on query type
        weights = self.WEIGHTS.get(parsed.query_type, self.WEIGHTS["style"])

        # Generate channel-specific prompts
        prompts = parsed.prompts

        # Encode query prompts for each channel
        query_vecs = {}
        for channel in ["global", "scene", "action"]:
            prompt = prompts.get(channel, query)
            query_vecs[channel] = self.encoder.encode_text(prompt)

        # Stage 1: Recall on global channel → top-50
        recall_k = max(top_k * 5, 50)
        global_results = self.store.search_channel("global", query_vecs["global"], top_k=recall_k)

        # Build candidate set
        candidate_ids = [img_id for img_id, _ in global_results]
        global_scores = {img_id: score for img_id, score in global_results}

        # Stage 2: Score all candidates across all channels
        # Search scene and action channels with larger top_k to cover candidates
        scene_results = self.store.search_channel("scene", query_vecs["scene"], top_k=recall_k)
        action_results = self.store.search_channel("action", query_vecs["action"], top_k=recall_k)

        scene_scores = {img_id: score for img_id, score in scene_results}
        action_scores = {img_id: score for img_id, score in action_results}

        # Composite scoring with dynamic weights
        scored = []
        for img_id in candidate_ids:
            s_global = global_scores.get(img_id, 0.0)
            s_scene = scene_scores.get(img_id, 0.0)
            s_action = action_scores.get(img_id, 0.0)

            # Scene threshold filtering
            if s_scene < self.SCENE_THRESHOLD and parsed.environment:
                continue

            final_score = (
                weights["global"] * s_global +
                weights["scene"] * s_scene +
                weights["action"] * s_action
            )

            scored.append({
                "image_id": img_id,
                "image_path": self.get_image_path(img_id),
                "score": final_score,
                "channel_scores": {
                    "global": s_global,
                    "scene": s_scene,
                    "action": s_action,
                },
            })

        # Sort by final score and return top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
