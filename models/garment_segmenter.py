"""SegFormer B2 Clothes garment segmentation wrapper.

Segments clothing into 18 fine-grained classes:
upper-clothes, pants, skirt, dress, belt, shoes, bag, hat, hair,
skin, face, left-arm, right-arm, left-leg, right-leg, left-shoe,
right-shoe, scarf, sunglasses, etc.

Model: mattmdjaga/segformer_b2_clothes on HuggingFace.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# SegFormer B2 Clothes label mapping
LABEL_NAMES = [
    "background", "hat", "hair", "sunglasses", "upper-clothes",
    "skirt", "pants", "dress", "belt", "left-shoe", "right-shoe",
    "face", "left-arm", "right-arm", "left-leg", "right-leg",
    "socks", "gloves", "tie", "scarf",
]

# Map garment regions to simplified categories for retrieval
REGION_MAPPING = {
    "upper-clothes": "upper",
    "dress": "dress",
    "pants": "lower",
    "skirt": "lower",
    "hat": "headwear",
    "tie": "accessory",
    "scarf": "accessory",
    "belt": "accessory",
    "sunglasses": "accessory",
    "gloves": "accessory",
    "bag": "accessory",
    "left-shoe": "footwear",
    "right-shoe": "footwear",
    "socks": "footwear",
}

# Labels we care about for garment region extraction
GARMENT_LABELS = {
    "upper-clothes", "pants", "skirt", "dress", "hat",
    "tie", "scarf", "belt", "sunglasses", "gloves",
    "left-shoe", "right-shoe", "socks", "bag",
}


class GarmentSegmenter:
    """SegFormer B2 Clothes wrapper for garment segmentation."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = "mattmdjaga/segformer_b2_clothes"

        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model = self.model.to(device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment an image into garment regions.

        Args:
            image: numpy array (H, W, 3), uint8.

        Returns:
            Dict mapping label name to binary mask (H, W) boolean array.
            Only includes detected garment labels.
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes, H/scale, W/scale)

        # Upsample to original size
        logits = torch.nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
        predicted = logits.argmax(dim=1)[0].cpu().numpy()

        # Extract masks per label
        masks = {}
        for label_idx, label_name in enumerate(LABEL_NAMES):
            if label_name in GARMENT_LABELS:
                mask = predicted == label_idx
                if mask.any():
                    masks[label_name] = mask

        return masks

    def segment_from_path(self, image_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Segment from image file path.

        Returns:
            Tuple of (masks_dict, image_array).
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        masks = self.segment(image)
        return masks, image

    def get_region_crops(self, image: np.ndarray, masks: Dict[str, np.ndarray], padding_ratio: float = 0.05) -> Dict[str, np.ndarray]:
        """Crop garment regions from image using segmentation masks.

        Args:
            image: numpy array (H, W, 3).
            masks: Dict from segment().
            padding_ratio: Padding around bbox.

        Returns:
            Dict mapping region category (upper, lower, etc.) to cropped image.
        """
        from utils.image_utils import crop_region_from_mask

        crops = {}
        for label_name, mask in masks.items():
            region_cat = REGION_MAPPING.get(label_name, label_name)
            if region_cat not in crops:  # Only keep first instance per category
                crop = crop_region_from_mask(image, mask, padding_ratio)
                if crop.size > 0:
                    crops[region_cat] = crop

        return crops
