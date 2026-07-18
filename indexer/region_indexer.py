"""V2: Region-level garment segmentation indexer.

Uses SegFormer B2 Clothes to segment garments, then embeds each region
separately with Marqo-FashionSigLIP. Also stores global and scene embeddings.

This directly addresses compositionality: "red" is bound to the upper garment
region, "blue" to the lower garment region. Global embeddings cannot achieve this.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from indexer.base_indexer import BaseIndexer
from models.clip_models import MarqoFashionSigLIPWrapper
from models.garment_segmenter import GarmentSegmenter
from models.person_detector import PersonDetector
from utils.vector_store import MultiVectorStore
from utils.image_utils import (
    image_id_from_path, load_image, mask_humans
)


class RegionIndexer(BaseIndexer):
    """V2: Region-level indexer with per-garment embeddings."""

    # Channels: global + scene + per-region types
    BASE_CHANNELS = ["global", "scene"]
    REGION_CHANNELS = ["upper", "lower", "dress", "headwear", "footwear", "accessory"]
    ALL_CHANNELS = BASE_CHANNELS + REGION_CHANNELS

    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        super().__init__(version_name="v2", embedding_dim=768, data_dir=data_dir, output_dir=output_dir)
        self.clip = MarqoFashionSigLIPWrapper(device=device)
        self.segmenter = GarmentSegmenter(device=device)
        self.detector = PersonDetector(conf_threshold=0.3, iou_threshold=0.5)

    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract per-region embeddings for each image.

        Returns:
            {image_id: {channel_name: np.ndarray(512,)}}
            Channels: global, scene, upper, lower, dress, headwear, footwear, accessory
            Missing regions use zero vectors.
        """
        features = {}
        zero_vec = np.zeros(self.embedding_dim, dtype=np.float32)

        for path in tqdm(image_paths, desc=f"[{self.version_name}] Extracting features"):
            img_id = image_id_from_path(path)
            image = np.array(load_image(path))

            # v_global: embed full image
            v_global = self.clip.encode_image_from_array(image)

            # Detect persons for scene masking
            boxes = self.detector.detect(image)
            if boxes:
                masked_image = mask_humans(image, boxes, padding=5)
                v_scene = self.clip.encode_image_from_array(masked_image)
            else:
                v_scene = v_global.copy()

            # Segment garments
            masks = self.segmenter.segment(image)
            region_crops = self.segmenter.get_region_crops(image, masks)

            # Embed each region crop
            region_embeddings = {}
            for channel in self.REGION_CHANNELS:
                if channel in region_crops:
                    region_embeddings[channel] = self.clip.encode_image_from_array(region_crops[channel])
                else:
                    region_embeddings[channel] = zero_vec.copy()

            # Assemble all channels
            features[img_id] = {
                "global": v_global,
                "scene": v_scene,
                **region_embeddings,
            }

        return features

    def build_index(self, features: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Build multi-vector FAISS index with all channels."""
        image_ids = sorted(features.keys())

        store = MultiVectorStore(dim=self.embedding_dim, channel_names=self.ALL_CHANNELS, metric="cosine")
        store.build(features, image_ids)
        store.save(str(self.output_dir))

        # Save metadata
        image_paths = getattr(self, '_image_paths', [])
        meta = {
            "image_paths": image_paths,
            "version": self.version_name,
            "channels": self.ALL_CHANNELS,
            "model": "Marqo-FashionSigLIP + SegFormer B2 + YOLOv8-nano",
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
