# Fashionista: Multimodal Fashion & Context Retrieval

An intelligent fashion image retrieval system that handles multi-attribute natural language queries involving clothing type, color, environment, and style. Built as a multi-version pipeline comparing progressive ML architectures against a vanilla CLIP baseline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download Fashionpedia dataset
bash scripts/download_data.sh

# Index images (choose version: v0a, v0b, v1, v2, v3, v4)
python scripts/index.py --version v0a --data_dir data/fashionpedia --output_dir indexes

# Retrieve images
python scripts/retrieve.py --query "A person in a bright yellow raincoat" --version v0a --top_k 10

# Evaluate all versions
python scripts/evaluate.py --all

# Launch web UI
python app/gradio_ui.py
```

## Versions

| Version | Approach | Key Innovation |
|---------|----------|----------------|
| V0a | Vanilla CLIP ViT-B/32 | Baseline |
| V0b | Marqo-FashionSigLIP | Domain-specific fashion CLIP |
| V1 | Multi-vector decomposition | Global + Scene + Action embeddings |
| V2 | Region-level segmentation | Per-garment-region embeddings |
| V3 | VLM captioning + dual-path | Florence-2 + BLIP captions |
| V4 | Hybrid ensemble | Multi-channel fusion + LABCLIP |

## Architecture

```
Indexer Pipeline:  Image → Feature Extraction → FAISS Vector Storage
Retriever Pipeline: Query → Parse → Embed → Vector Search → Re-rank → Top-K Images
```

## Dataset

Uses [Fashionpedia](https://github.com/cvdfoundation/fashionpedia) — 48K+ images with segmentation masks and 294 fine-grained attributes.

## Evaluation

5 query categories: Attribute Specific, Contextual, Complex Semantic, Style Inference, Compositional. 100 test queries total with Precision@K, Recall@K, and MRR metrics.
