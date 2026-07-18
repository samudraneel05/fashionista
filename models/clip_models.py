"""CLIP and Fashion-CLIP model wrappers for image and text encoding."""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from pathlib import Path


class CLIPWrapper:
    """Wrapper for OpenAI CLIP ViT-B/32 (vanilla baseline)."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = "ViT-B/32"

        import clip
        self.model, self.preprocess = clip.load(self.model_name, device=device)
        self.model.eval()
        self.embedding_dim = 512

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode images into L2-normalized embeddings.

        Args:
            image_paths: List of image file paths.
            batch_size: Batch size for inference.

        Returns:
            numpy array of shape (N, 512), L2-normalized.
        """
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    images.append(torch.zeros(3, 224, 224))

            image_input = torch.stack(images).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Encode a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into L2-normalized embeddings.

        Args:
            texts: Single string or list of strings.

        Returns:
            numpy array of shape (N, 512) or (512,) for single text.
        """
        import clip
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        result = features.cpu().numpy()
        return result.flatten() if single else result

    def encode_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Encode an image from a numpy array (H, W, 3)."""
        image = Image.fromarray(image_array)
        return self.encode_image(image)


class MarqoFashionSigLIPWrapper:
    """Wrapper for Marqo-FashionSigLIP — SOTA fashion-specific CLIP model.

    Fine-tuned from ViT-B-16-SigLIP with Generalized Contrastive Learning (GCL)
    on categories, colors, materials, styles, and fine-details.
    Shows 57% improvement over FashionCLIP 2.0 on 6 fashion benchmarks.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = "Marqo/marqo-fashionSigLIP"

        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        self.embedding_dim = 512

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode images into L2-normalized embeddings."""
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    images.append(torch.zeros(3, 224, 224))

            image_input = torch.stack(images).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Encode a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into L2-normalized embeddings."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        result = features.cpu().numpy()
        return result.flatten() if single else result

    def encode_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Encode an image from a numpy array (H, W, 3)."""
        image = Image.fromarray(image_array)
        return self.encode_image(image)
