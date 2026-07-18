"""V1: Multi-vector semantic decomposition indexer.

Generates 3 embeddings per image:
- v_global: CLIP/Fashion-CLIP on full image (overall semantic)
- v_scene: CLIP on image with humans masked out (pure environment)
- v_action: CLIP on cropped person (fashion-focused)

Inspired by FashionGlance (previous intern, selected).
Uses YOLOv8-nano for person detection, Marqo-FashionSigLIP for embeddings.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from indexer.base_indexer import BaseIndexer
from models.clip_models import MarqoFashionSigLIPWrapper
from models.person_detector import PersonDetector
from utils.vector_store import MultiVectorStore
from utils.image_utils import (
    image_id_from_path, load_image, mask_humans, crop_largest_person
)


class MultiVectorIndexer(BaseIndexer):
    """V1: Multi-vector indexer with global, scene, and action embeddings."""

    CHANNEL_NAMES = ["global", "scene", "action"]

    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        super().__init__(version_name="v1", embedding_dim=512, data_dir=data_dir, output_dir=output_dir)
        self.clip = MarqoFashionSigLIPWrapper(device=device)
        self.detector = PersonDetector(conf_threshold=0.3, iou_threshold=0.5)

    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract 3 embeddings per image: global, scene, action.

        Returns:
            {image_id: {"global": np.ndarray(512,), "scene": np.ndarray(512,), "action": np.ndarray(512,)}}
        """
        features = {}

        for path in tqdm(image_paths, desc=f"[{self.version_name}] Extracting features"):
            img_id = image_id_from_path(path)
            image = np.array(load_image(path))

            # v_global: embed full image
            v_global = self.clip.encode_image_from_array(image)

            # Detect persons
            boxes = self.detector.detect(image)

            if boxes:
                # v_scene: mask humans, embed background
                masked_image = mask_humans(image, boxes, padding=5)
                v_scene = self.clip.encode_image_from_array(masked_image)

                # v_action: crop largest person, embed
                person_crop = crop_largest_person(image, boxes, padding_ratio=0.1)
                if person_crop.size > 0:
                    v_action = self.clip.encode_image_from_array(person_crop)
                else:
                    v_action = v_global.copy()
            else:
                # No person detected — scene = global, action = global
                v_scene = v_global.copy()
                v_action = v_global.copy()

            features[img_id] = {
                "global": v_global,
                "scene": v_scene,
                "action": v_action,
            }

        return features

    def build_index(self, features: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Build multi-vector FAISS index."""
        image_ids = sorted(features.keys())

        store = MultiVectorStore(dim=self.embedding_dim, channel_names=self.CHANNEL_NAMES, metric="cosine")
        store.build(features, image_ids)
        store.save(str(self.output_dir))

        # Save metadata
        image_paths = getattr(self, '_image_paths', [])
        meta = {"image_paths": image_paths, "version": self.version_name,
                "channels": self.CHANNEL_NAMES, "model": "Marqo-FashionSigLIP + YOLOv8-nano"}
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
