"""YOLOv8 person detection wrapper."""

import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class PersonDetector:
    """YOLOv8-nano wrapper for person detection.

    Detects persons in images and returns bounding boxes.
    Used for human masking (scene separation) and person cropping (action embedding).
    """

    def __init__(self, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
        """Initialize YOLOv8-nano.

        Args:
            conf_threshold: Detection confidence threshold.
            iou_threshold: NMS IoU threshold.
        """
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = 0  # COCO class 0 = person

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """Detect persons in an image.

        Args:
            image: numpy array (H, W, 3), uint8.

        Returns:
            List of [x1, y1, x2, y2] bounding boxes for detected persons.
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append([float(x) for x in xyxy])

        return boxes

    def detect_from_path(self, image_path: str) -> Tuple[List[List[float]], np.ndarray]:
        """Detect persons from an image file path.

        Returns:
            Tuple of (boxes, image_array).
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        boxes = self.detect(image)
        return boxes, image

    def detect_batch(self, image_paths: List[str]) -> List[Tuple[List[List[float]], np.ndarray]]:
        """Detect persons in a batch of images."""
        results = []
        for path in image_paths:
            results.append(self.detect_from_path(path))
        return results
