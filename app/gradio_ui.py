"""Gradio web UI for fashion image retrieval.

Provides a visual interface for querying the retrieval system with
any of the available versions (V0a–V4). Shows retrieved images with
scores and per-channel breakdowns.

Usage:
    python app/gradio_ui.py --version v4 --index_dir indexes --data_dir data/fashionpedia
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from PIL import Image


def get_retriever(version: str, index_dir: str, data_dir: str, device: str = None):
    """Get retriever for the specified version."""
    version_index_dir = str(Path(index_dir) / version)

    if version in ("v0a", "v0b"):
        from retriever.clip_retriever import CLIPRetriever
        return CLIPRetriever(version_name=version, index_dir=version_index_dir, data_dir=data_dir, device=device)
    elif version == "v1":
        from retriever.multi_vector_retriever import MultiVectorRetriever
        return MultiVectorRetriever(index_dir=version_index_dir, data_dir=data_dir, device=device)
    elif version == "v2":
        from retriever.region_retriever import RegionRetriever
        return RegionRetriever(index_dir=version_index_dir, data_dir=data_dir, device=device)
    elif version == "v3":
        from retriever.vlm_retriever import VLMRetriever
        return VLMRetriever(index_dir=version_index_dir, data_dir=data_dir, device=device)
    elif version == "v4":
        from retriever.hybrid_retriever import HybridRetriever
        return HybridRetriever(index_dir=version_index_dir, data_dir=data_dir, device=device)
    else:
        raise ValueError(f"Unknown version: {version}")


# Global retrievers dict — loaded on launch
RETRIEVERS: Dict[str, Any] = {}


def search_images(query: str, version: str, top_k: int) -> List[tuple]:
    """Search for images and return gallery results."""
    if version not in RETRIEVERS:
        return [], f"Version {version} not loaded."

    retriever = RETRIEVERS[version]
    results = retriever.retrieve(query, top_k=top_k)

    gallery_images = []
    score_text = f"**Query:** {query}\n**Version:** {version}\n**Top-{top_k} Results:**\n\n"

    for i, result in enumerate(results):
        img_path = result.get("image_path")
        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert("RGB")
            score = result["score"]
            label = f"#{i+1} {result['image_id']} (score: {score:.4f})"
            gallery_images.append((img, label))

            score_text += f"**#{i+1}** `{result['image_id']}` — Score: {score:.4f}\n"

            # Show channel scores if available
            if "channel_scores" in result:
                for channel, ch_score in result["channel_scores"].items():
                    score_text += f"  - {channel}: {ch_score:.4f}\n"

            # Show caption if available
            if result.get("caption"):
                score_text += f"  - Caption: {result['caption'][:100]}...\n"

            score_text += "\n"
        else:
            score_text += f"**#{i+1}** `{result['image_id']}` — Score: {result['score']:.4f} (image not found)\n\n"

    if not gallery_images:
        score_text += "No results found.\n"

    return gallery_images, score_text


def create_ui(
    versions: List[str],
    index_dir: str,
    data_dir: str,
    device: str = None,
    default_queries: List[str] = None,
) -> gr.Blocks:
    """Create the Gradio UI."""

    if default_queries is None:
        default_queries = [
            "A person in a bright yellow raincoat.",
            "Professional business attire inside a modern office.",
            "Someone wearing a blue shirt sitting on a park bench.",
            "Casual weekend outfit for a city walk.",
            "A red tie and a white shirt in a formal setting.",
        ]

    with gr.Blocks(title="Fashionista: Fashion & Context Retrieval", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🔍 Fashionista: Multimodal Fashion & Context Retrieval")
        gr.Markdown("Search for fashion images using natural language descriptions.")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="e.g., A person in a bright yellow raincoat",
                    lines=2,
                )
                with gr.Row():
                    version_selector = gr.Dropdown(
                        choices=versions,
                        value=versions[0],
                        label="Model Version",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Top-K Results",
                    )
                search_btn = gr.Button("🔍 Search", variant="primary")

                # Quick query examples
                gr.Markdown("### Example Queries")
                for q in default_queries:
                    gr.Examples(
                        examples=[[q]],
                        inputs=[query_input],
                    )

            with gr.Column(scale=2):
                score_output = gr.Markdown(label="Scores & Details")

        gallery = gr.Gallery(
            label="Retrieved Images",
            columns=5,
            height=400,
            show_label=True,
        )

        search_btn.click(
            fn=search_images,
            inputs=[query_input, version_selector, top_k_slider],
            outputs=[gallery, score_output],
        )

        query_input.submit(
            fn=search_images,
            inputs=[query_input, version_selector, top_k_slider],
            outputs=[gallery, score_output],
        )

    return ui


def main():
    parser = argparse.ArgumentParser(description="Launch Fashionista Gradio UI")
    parser.add_argument("--versions", nargs="+", default=["v0a"],
                        help="Versions to load (v0a, v0b, v1, v2, v3, v4)")
    parser.add_argument("--index_dir", type=str, default="indexes")
    parser.add_argument("--data_dir", type=str, default="data/fashionpedia")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Load retrievers
    for version in args.versions:
        try:
            print(f"Loading {version}...")
            RETRIEVERS[version] = get_retriever(version, args.index_dir, args.data_dir, args.device)
            print(f"  ✓ {version} loaded")
        except Exception as e:
            print(f"  ✗ {version} failed: {e}")

    if not RETRIEVERS:
        print("No retrievers loaded. Exiting.")
        sys.exit(1)

    ui = create_ui(
        versions=list(RETRIEVERS.keys()),
        index_dir=args.index_dir,
        data_dir=args.data_dir,
        device=args.device,
    )
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
