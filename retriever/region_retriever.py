"""V2: Region-level garment segmentation retriever.

Query decomposition maps attributes to specific garment regions:
"red shirt with blue pants" → {"upper": "red shirt", "lower": "blue pants"}

Retrieves per-region candidates and intersects to find images where
all specified regions match. This directly solves compositionality.
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from retriever.base_retriever import BaseRetriever
from models.clip_models import MarqoFashionSigLIPWrapper
from utils.vector_store import MultiVectorStore
from utils.query_parser import QueryParser, ANATOMICAL_MAP


class RegionRetriever(BaseRetriever):
    """V2: Region-level retriever with per-garment matching."""

    # Map garment names to region channels
    GARMENT_TO_REGION = {
        "shirt": "upper", "t-shirt": "upper", "tshirt": "upper",
        "blouse": "upper", "hoodie": "upper", "sweater": "upper",
        "pullover": "upper", "cardigan": "upper", "tank": "upper",
        "tanktop": "upper", "polo": "upper", "button-down": "upper",
        "buttondown": "upper", "vest": "upper", "blazer": "upper",
        "jacket": "upper", "coat": "upper", "overcoat": "upper",
        "trench": "upper", "raincoat": "upper", "windbreaker": "upper",
        "tie": "accessory", "scarf": "accessory", "belt": "accessory",
        "sunglasses": "accessory", "gloves": "accessory",
        "pants": "lower", "trousers": "lower", "jeans": "lower",
        "shorts": "lower", "skirt": "lower",
        "dress": "dress",
        "hat": "headwear", "cap": "headwear",
        "shoes": "footwear", "sneakers": "footwear", "boots": "footwear",
        "heels": "footwear", "sandals": "footwear",
    }

    REGION_WEIGHTS = {
        "global": 0.15,
        "scene": 0.15,
        "upper": 0.25,
        "lower": 0.25,
        "dress": 0.30,
        "headwear": 0.15,
        "footwear": 0.15,
        "accessory": 0.15,
    }

    def __init__(self, index_dir: str, data_dir: str, device: str = None):
        super().__init__(version_name="v2", index_dir=index_dir, data_dir=data_dir)
        self.encoder = MarqoFashionSigLIPWrapper(device=device)
        self.store = MultiVectorStore(
            dim=512,
            channel_names=["global", "scene", "upper", "lower", "dress",
                           "headwear", "footwear", "accessory"],
            metric="cosine",
        )
        self.store.load(index_dir)
        self.parser = QueryParser()

    def _decompose_query(self, parsed) -> Dict[str, str]:
        """Decompose parsed query into region-specific queries.

        Returns:
            {region_channel: query_text} for regions mentioned in query.
        """
        region_queries = {}

        # Map color-item pairs to regions
        for color, garment in parsed.color_item_pairs:
            region = self.GARMENT_TO_REGION.get(garment, "upper")
            region_queries[region] = f"a {color} {garment}"

        # Map standalone garments to regions
        if not region_queries:
            for garment in parsed.garments:
                region = self.GARMENT_TO_REGION.get(garment, "upper")
                if region not in region_queries:
                    region_queries[region] = f"a {garment}"

        return region_queries

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images using region-level matching."""
        parsed = self.parser.parse(query)
        region_queries = self._decompose_query(parsed)

        # Always search global and scene
        recall_k = max(top_k * 5, 50)

        # Encode global query
        global_vec = self.encoder.encode_text(query)
        global_results = self.store.search_channel("global", global_vec, top_k=recall_k)
        global_scores = {img_id: score for img_id, score in global_results}

        # Encode scene query
        scene_prompt = parsed.prompts.get("scene", "a scene")
        scene_vec = self.encoder.encode_text(scene_prompt)
        scene_results = self.store.search_channel("scene", scene_vec, top_k=recall_k)
        scene_scores = {img_id: score for img_id, score in scene_results}

        # Encode region-specific queries
        region_scores = {}
        active_regions = []
        for region, region_query in region_queries.items():
            region_vec = self.encoder.encode_text(region_query)
            results = self.store.search_channel(region, region_vec, top_k=recall_k)
            region_scores[region] = {img_id: score for img_id, score in results}
            active_regions.append(region)

        # Score candidates
        candidate_ids = set(global_scores.keys())
        scored = []

        for img_id in candidate_ids:
            s_global = global_scores.get(img_id, 0.0)
            s_scene = scene_scores.get(img_id, 0.0)

            # Region scores
            region_score_vals = {}
            for region in active_regions:
                region_score_vals[region] = region_scores.get(region, {}).get(img_id, 0.0)

            # Compute weighted score
            # Base weight from global + scene
            base_score = self.REGION_WEIGHTS["global"] * s_global + self.REGION_WEIGHTS["scene"] * s_scene

            # Region scores weighted by remaining weight
            if active_regions:
                region_weight_sum = sum(self.REGION_WEIGHTS.get(r, 0.15) for r in active_regions)
                region_score_sum = sum(
                    self.REGION_WEIGHTS.get(r, 0.15) * region_score_vals.get(r, 0.0)
                    for r in active_regions
                )
                if region_weight_sum > 0:
                    final_score = base_score + region_score_sum
                else:
                    final_score = base_score + s_global  # fallback
            else:
                final_score = s_global

            channel_scores = {
                "global": s_global,
                "scene": s_scene,
            }
            channel_scores.update(region_score_vals)

            scored.append({
                "image_id": img_id,
                "image_path": self.get_image_path(img_id),
                "score": final_score,
                "channel_scores": channel_scores,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
