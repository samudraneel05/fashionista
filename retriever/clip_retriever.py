"""V0a/V0b: Single-vector CLIP retriever.

Works with both vanilla CLIP (V0a) and Marqo-FashionSigLIP (V0b) indexes.
Simple text → embedding → FAISS search → top-k results.
"""

from typing import List, Dict, Any
from pathlib import Path

from retriever.base_retriever import BaseRetriever
from utils.vector_store import VectorStore


class CLIPRetriever(BaseRetriever):
    """Single-vector CLIP retriever for V0a and V0b."""

    def __init__(self, version_name: str, index_dir: str, data_dir: str, device: str = None):
        super().__init__(version_name=version_name, index_dir=index_dir, data_dir=data_dir)
        self.device = device
        self.store = VectorStore(dim=512, metric="cosine")
        self.store.load(str(Path(index_dir) / "global"))

        # Load the appropriate model for text encoding
        if version_name == "v0a":
            from models.clip_models import CLIPWrapper
            self.encoder = CLIPWrapper(device=device)
        elif version_name == "v0b":
            from models.clip_models import MarqoFashionSigLIPWrapper
            self.encoder = MarqoFashionSigLIPWrapper(device=device)
        else:
            raise ValueError(f"Unknown version: {version_name}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k images for a query."""
        # Encode query text
        query_vec = self.encoder.encode_text(query)

        # Search FAISS index
        results = self.store.search(query_vec, top_k=top_k)

        # Format output
        output = []
        for image_id, score in results:
            image_path = self.get_image_path(image_id)
            output.append({
                "image_id": image_id,
                "image_path": image_path,
                "score": score,
            })

        return output
