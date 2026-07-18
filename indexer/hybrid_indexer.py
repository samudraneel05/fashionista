"""V4: Hybrid ensemble indexer.

Combines V1 (multi-vector), V2 (region-level), and V3 (VLM captioning)
into a single 6-vector representation per image:
- v_global: Marqo on full image
- v_scene: Marqo on masked background
- v_action: Marqo on person crop
- v_region_upper: Marqo on upper garment crop
- v_region_lower: Marqo on lower garment crop
- v_caption: BGE on Florence-2 + BLIP combined caption

Also applies LABCLIP linear transformation to text embeddings at retrieval time.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from indexer.base_indexer import BaseIndexer
from models.clip_models import MarqoFashionSigLIPWrapper
from models.person_detector import PersonDetector
from models.garment_segmenter import GarmentSegmenter
from models.vlm_captioner import VLMCaptioner
from utils.vector_store import MultiVectorStore, VectorStore
from utils.image_utils import (
    image_id_from_path, load_image, mask_humans, crop_largest_person,
    crop_region_from_mask,
)


class HybridIndexer(BaseIndexer):
    """V4: Hybrid ensemble indexer combining all approaches."""

    CLIP_CHANNELS = ["global", "scene", "action", "region_upper", "region_lower"]
    CAPTION_DIM = 1024  # BGE-large dimension

    def __init__(self, data_dir: str, output_dir: str, device: str = None):
        super().__init__(version_name="v4", embedding_dim=768, data_dir=data_dir, output_dir=output_dir)
        self.device = device
        self.clip = MarqoFashionSigLIPWrapper(device=device)
        self.detector = PersonDetector(conf_threshold=0.3, iou_threshold=0.5)
        self.segmenter = GarmentSegmenter(device=device)
        self.vlm = VLMCaptioner(device=device)

        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

        self.captions: Dict[str, Dict[str, str]] = {}

    def extract_features(self, image_paths: List[str], batch_size: int = 16) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract all 6 vectors per image.

        Returns:
            {image_id: {channel: np.ndarray}}
            CLIP channels: 512-d, caption channel: 1024-d
        """
        clip_features = {}
        captions = {}
        zero_vec = np.zeros(512, dtype=np.float32)

        for path in tqdm(image_paths, desc=f"[{self.version_name}] Full feature extraction"):
            img_id = image_id_from_path(path)
            image = np.array(load_image(path))

            # --- CLIP-based embeddings (512-d) ---
            v_global = self.clip.encode_image_from_array(image)

            # Person detection
            boxes = self.detector.detect(image)
            if boxes:
                masked_image = mask_humans(image, boxes, padding=5)
                v_scene = self.clip.encode_image_from_array(masked_image)

                person_crop = crop_largest_person(image, boxes, padding_ratio=0.1)
                v_action = self.clip.encode_image_from_array(person_crop) if person_crop.size > 0 else v_global.copy()
            else:
                v_scene = v_global.copy()
                v_action = v_global.copy()

            # Garment segmentation for region crops
            masks = self.segmenter.segment(image)
            region_crops = self.segmenter.get_region_crops(image, masks)

            v_region_upper = self.clip.encode_image_from_array(region_crops["upper"]) if "upper" in region_crops else zero_vec.copy()
            v_region_lower = self.clip.encode_image_from_array(region_crops["lower"]) if "lower" in region_crops else zero_vec.copy()

            clip_features[img_id] = {
                "global": v_global,
                "scene": v_scene,
                "action": v_action,
                "region_upper": v_region_upper,
                "region_lower": v_region_lower,
            }

            # --- VLM captioning ---
            try:
                caps = self.vlm.caption(image)
                captions[img_id] = caps
            except Exception as e:
                print(f"Error captioning {path}: {e}")
                captions[img_id] = {"florence": "", "blip": "", "combined": ""}

        self.captions = captions

        # --- BGE text embeddings (1024-d) ---
        image_ids = sorted(captions.keys())
        texts = [captions[img_id]["combined"] for img_id in image_ids]

        print(f"[{self.version_name}] Encoding {len(texts)} captions with BGE-large...")
        caption_embeddings = self.text_encoder.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Add caption embeddings to features
        for img_id, emb in zip(image_ids, caption_embeddings):
            clip_features[img_id]["caption"] = emb.astype(np.float32)

        return clip_features

    def build_index(self, features: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Build two FAISS indexes: multi-vector (CLIP channels) + single (caption)."""
        image_ids = sorted(features.keys())

        # Multi-vector index for CLIP channels (512-d)
        clip_store = MultiVectorStore(
            dim=512,
            channel_names=self.CLIP_CHANNELS,
            metric="cosine",
        )
        clip_features = {
            img_id: {ch: features[img_id][ch] for ch in self.CLIP_CHANNELS}
            for img_id in image_ids
        }
        clip_store.build(clip_features, image_ids)
        clip_store.save(str(self.output_dir / "clip_vectors"))

        # Single-vector index for caption embeddings (1024-d)
        caption_store = VectorStore(dim=self.CAPTION_DIM, metric="cosine")
        caption_embeddings = np.stack([features[img_id]["caption"] for img_id in image_ids])
        caption_store.build(caption_embeddings, image_ids)
        caption_store.save(str(self.output_dir / "caption"))

        # Save captions
        with open(self.output_dir / "captions.json", "w") as f:
            json.dump(self.captions, f, indent=2)

        # Save metadata
        image_paths = getattr(self, '_image_paths', [])
        meta = {
            "image_paths": image_paths,
            "version": self.version_name,
            "clip_channels": self.CLIP_CHANNELS,
            "caption_dim": self.CAPTION_DIM,
            "model": "Marqo-FashionSigLIP + SegFormer B2 + YOLOv8 + Florence-2 + BLIP + BGE-large + LABCLIP",
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
