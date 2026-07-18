"""V4: Hybrid ensemble retriever.

Multi-channel fusion with query-adaptive weighting:
- Region vectors for compositional queries (upper/lower binding)
- Scene vector for contextual queries (environment)
- Caption similarity for style/semantic queries
- Global vector for recall

Also applies LABCLIP linear transformation to text embeddings for better
attribute-object binding (fixes CLIP's bag-of-words cross-modal behavior).

Three-stage retrieval:
1. HNSW recall on v_global → top-100
2. Multi-channel scoring with dynamic weights
3. Cross-encoder re-ranking on top-10
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from retriever.base_retriever import BaseRetriever
from models.clip_models import MarqoFashionSigLIPWrapper, CLIPWrapper
from utils.vector_store import MultiVectorStore, VectorStore
from utils.query_parser import QueryParser


class HybridRetriever(BaseRetriever):
    """V4: Hybrid ensemble retriever with multi-channel fusion."""

    # Dynamic weight profiles
    WEIGHTS = {
        "compositional": {
            "global": 0.10, "scene": 0.05, "action": 0.20,
            "region_upper": 0.25, "region_lower": 0.25, "caption": 0.15,
        },
        "attribute": {
            "global": 0.15, "scene": 0.05, "action": 0.30,
            "region_upper": 0.20, "region_lower": 0.10, "caption": 0.20,
        },
        "contextual": {
            "global": 0.20, "scene": 0.40, "action": 0.10,
            "region_upper": 0.05, "region_lower": 0.05, "caption": 0.20,
        },
        "complex": {
            "global": 0.20, "scene": 0.20, "action": 0.20,
            "region_upper": 0.15, "region_lower": 0.10, "caption": 0.15,
        },
        "style": {
            "global": 0.25, "scene": 0.10, "action": 0.15,
            "region_upper": 0.05, "region_lower": 0.05, "caption": 0.40,
        },
    }

    GARMENT_TO_REGION = {
        "shirt": "region_upper", "t-shirt": "region_upper", "tshirt": "region_upper",
        "blouse": "region_upper", "hoodie": "region_upper", "sweater": "region_upper",
        "pullover": "region_upper", "cardigan": "region_upper", "tank": "region_upper",
        "tanktop": "region_upper", "polo": "region_upper", "button-down": "region_upper",
        "buttondown": "region_upper", "vest": "region_upper", "blazer": "region_upper",
        "jacket": "region_upper", "coat": "region_upper", "overcoat": "region_upper",
        "trench": "region_upper", "raincoat": "region_upper", "windbreaker": "region_upper",
        "tie": "region_upper", "scarf": "region_upper",
        "pants": "region_lower", "trousers": "region_lower", "jeans": "region_lower",
        "shorts": "region_lower", "skirt": "region_lower",
    }

    SCENE_THRESHOLD = 0.1

    def __init__(self, index_dir: str, data_dir: str, device: str = None):
        super().__init__(version_name="v4", index_dir=index_dir, data_dir=data_dir)
        self.device = device
        self.parser = QueryParser()

        # Marqo for CLIP channel encoding
        self.marqo = MarqoFashionSigLIPWrapper(device=device)

        # CLIP for cross-modal re-ranking
        self.clip = CLIPWrapper(device=device)

        # BGE for caption encoding
        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

        # Load multi-vector CLIP index
        self.clip_store = MultiVectorStore(
            dim=768,
            channel_names=["global", "scene", "action", "region_upper", "region_lower"],
            metric="cosine",
        )
        self.clip_store.load(str(Path(index_dir) / "clip_vectors"))

        # Load caption index
        self.caption_store = VectorStore(dim=1024, metric="cosine")
        self.caption_store.load(str(Path(index_dir) / "caption"))

        # Load captions
        import json
        captions_path = Path(index_dir) / "captions.json"
        self.captions = {}
        if captions_path.exists():
            with open(captions_path, "r") as f:
                self.captions = json.load(f)

        # LABCLIP transform (identity if not trained)
        self.labclip_transform = None  # Will be loaded if available
        labclip_path = Path(index_dir) / "labclip_weights.npy"
        if labclip_path.exists():
            self.labclip_transform = np.load(str(labclip_path))

    def _apply_labclip(self, text_embedding: np.ndarray) -> np.ndarray:
        """Apply LABCLIP linear transformation to text embedding."""
        if self.labclip_transform is not None:
            return self.labclip_transform @ text_embedding
        return text_embedding

    def _get_region_queries(self, parsed) -> Dict[str, str]:
        """Decompose query into region-specific prompts."""
        region_queries = {}
        for color, garment in parsed.color_item_pairs:
            region = self.GARMENT_TO_REGION.get(garment, "region_upper")
            if region not in region_queries:
                region_queries[region] = f"a {color} {garment}"
        if not region_queries:
            for garment in parsed.garments:
                region = self.GARMENT_TO_REGION.get(garment, "region_upper")
                if region not in region_queries:
                    region_queries[region] = f"a {garment}"
        return region_queries

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images using three-stage hybrid search."""
        parsed = self.parser.parse(query)
        weights = self.WEIGHTS.get(parsed.query_type, self.WEIGHTS["style"])
        region_queries = self._get_region_queries(parsed)

        recall_k = max(top_k * 10, 100)

        # --- Encode queries for each channel ---

        # CLIP channels (with LABCLIP)
        clip_query_vecs = {}
        for channel in ["global", "scene", "action"]:
            prompt = parsed.prompts.get(channel, query)
            vec = self.marqo.encode_text(prompt)
            clip_query_vecs[channel] = self._apply_labclip(vec)

        # Region channels
        for region_channel, region_query in region_queries.items():
            vec = self.marqo.encode_text(region_query)
            clip_query_vecs[region_channel] = self._apply_labclip(vec)

        # Caption channel (BGE)
        caption_vec = self.text_encoder.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        # --- Stage 1: HNSW recall on global → top-100 ---
        global_results = self.clip_store.search_channel("global", clip_query_vecs["global"], top_k=recall_k)
        candidate_ids = [img_id for img_id, _ in global_results]

        if not candidate_ids:
            return []

        # --- Stage 2: Multi-channel scoring ---

        # Search all CLIP channels
        all_clip_scores = {}
        for channel in ["global", "scene", "action", "region_upper", "region_lower"]:
            if channel in clip_query_vecs:
                results = self.clip_store.search_channel(channel, clip_query_vecs[channel], top_k=recall_k)
                all_clip_scores[channel] = {img_id: score for img_id, score in results}
            else:
                all_clip_scores[channel] = {}

        # Search caption channel
        caption_results = self.caption_store.search(caption_vec, top_k=recall_k)
        caption_scores = {img_id: score for img_id, score in caption_results}

        # Score all candidates
        scored = []
        for img_id in candidate_ids:
            channel_scores = {}
            for channel in ["global", "scene", "action", "region_upper", "region_lower"]:
                channel_scores[channel] = all_clip_scores.get(channel, {}).get(img_id, 0.0)
            channel_scores["caption"] = caption_scores.get(img_id, 0.0)

            # Scene threshold filtering
            if parsed.environment and channel_scores["scene"] < self.SCENE_THRESHOLD:
                continue

            # Weighted fusion
            final_score = sum(weights.get(ch, 0.0) * channel_scores.get(ch, 0.0) for ch in weights)

            scored.append({
                "image_id": img_id,
                "image_path": self.get_image_path(img_id),
                "score": final_score,
                "channel_scores": channel_scores,
                "caption": self.captions.get(img_id, {}).get("combined", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # --- Stage 3: Cross-encoder re-ranking on top-10 ---
        top_candidates = scored[:max(top_k * 2, 20)]

        if len(top_candidates) > top_k:
            # CLIP cross-modal re-ranking
            clip_query_vec = self.clip.encode_text(query)
            for candidate in top_candidates:
                img_path = candidate.get("image_path")
                if img_path:
                    try:
                        img_vec = self.clip.encode_image(img_path)
                        cross_modal_score = float(np.dot(clip_query_vec, img_vec))
                        # Blend: 70% multi-channel score + 30% cross-modal
                        candidate["score"] = 0.7 * candidate["score"] + 0.3 * cross_modal_score
                        candidate["channel_scores"]["cross_modal"] = cross_modal_score
                    except Exception:
                        candidate["channel_scores"]["cross_modal"] = 0.0

            top_candidates.sort(key=lambda x: x["score"], reverse=True)

        return top_candidates[:top_k]
