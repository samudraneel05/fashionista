"""Image utilities: loading, masking, cropping, preprocessing."""

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict
from pathlib import Path


def load_image(path: str) -> Image.Image:
    """Load an image as RGB PIL Image."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def mask_humans(image: np.ndarray, boxes: List[List[float]], padding: int = 0) -> np.ndarray:
    """Mask out human bounding boxes in an image (set to black).

    Args:
        image: numpy array of shape (H, W, 3), uint8.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        padding: Extra padding around boxes to mask.

    Returns:
        Masked image (copy), same shape.
    """
    masked = image.copy()
    h, w = masked.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        masked[y1:y2, x1:x2] = 0
    return masked


def crop_person(image: np.ndarray, box: List[float], padding_ratio: float = 0.1) -> np.ndarray:
    """Crop a person from an image with padding.

    Args:
        image: numpy array of shape (H, W, 3), uint8.
        box: [x1, y1, x2, y2] bounding box.
        padding_ratio: Fraction of box size to add as padding.

    Returns:
        Cropped image as numpy array.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    return image[y1:y2, x1:x2]


def crop_largest_person(image: np.ndarray, boxes: List[List[float]], padding_ratio: float = 0.1) -> Optional[np.ndarray]:
    """Crop the largest person by bounding box area.

    Args:
        image: numpy array (H, W, 3).
        boxes: List of [x1, y1, x2, y2].
        padding_ratio: Padding fraction.

    Returns:
        Cropped image or None if no boxes.
    """
    if not boxes:
        return None

    # Find largest box by area
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    largest_idx = int(np.argmax(areas))
    return crop_person(image, boxes[largest_idx], padding_ratio)


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """Convert a binary segmentation mask to a bounding box [x1, y1, x2, y2]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 + 1), int(y2 + 1)]


def crop_region_from_mask(image: np.ndarray, mask: np.ndarray, padding_ratio: float = 0.05) -> np.ndarray:
    """Crop a region from an image using a segmentation mask.

    Args:
        image: numpy array (H, W, 3).
        mask: binary mask (H, W), True where region of interest.
        padding_ratio: Padding around bbox.

    Returns:
        Cropped image.
    """
    bbox = mask_to_bbox(mask)
    return crop_person(image, bbox, padding_ratio)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Apply a segmentation mask to an image.

    Args:
        image: numpy array (H, W, 3).
        mask: boolean mask (H, W), True where region of interest.
        inverse: If True, black out the masked region (keep background).
                 If False, black out everything except the masked region (keep region).

    Returns:
        Image with selected regions set to black.
    """
    result = image.copy()
    if inverse:
        result[mask] = 0
    else:
        result[~mask] = 0
    return result


def get_image_paths(data_dir: str, extensions: set = None) -> List[str]:
    """Get all image file paths from a directory tree."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
    data_dir = Path(data_dir)
    paths = sorted([
        str(p) for p in data_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])
    return paths


def image_id_from_path(path: str) -> str:
    """Extract image_id (filename without extension) from path."""
    return Path(path).stem
