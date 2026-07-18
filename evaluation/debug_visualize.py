"""Visual debugging: show retrieved images vs ground truth for each query.

For each query, displays a grid of top-10 retrieved images with green borders
for relevant (ground truth) hits and red borders for misses.
Saves grids as PNGs in evaluation/debug_plots/.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from evaluation.queries import get_base_queries, get_all_queries
from evaluation.ground_truth import GroundTruthGenerator


def visualize_query_results(
    version: str,
    queries: List[Dict],
    retriever,
    gt_gen: GroundTruthGenerator,
    data_dir: str,
    output_dir: str,
    top_k: int = 10,
):
    """For each query, save a grid of retrieved images with GT markers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for q in queries:
        query_text = q["query"]
        query_cat = q["category"]

        # Retrieve
        results = retriever.retrieve(query_text, top_k=top_k)
        retrieved_ids = [r["image_id"] for r in results]

        # Ground truth
        candidate_ids = [r["image_id"] for r in retriever.retrieve(query_text, top_k=50)]
        relevant_ids = gt_gen.get_ground_truth(q["id"], query_text, query_cat, candidate_ids)

        # Hits
        hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)

        # Build grid
        n_cols = min(top_k, 10)
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, 3))
        if n_cols == 1:
            axes = [axes]

        fig.suptitle(
            f"[{version}] {query_cat}: \"{query_text}\"\n"
            f"Hits: {hits}/{top_k} | GT pool: {len(relevant_ids)} images",
            fontsize=10, fontweight="bold"
        )

        for i, (ax, r) in enumerate(zip(axes, results)):
            img_id = r["image_id"]
            score = r.get("score", 0)

            # Find image path
            img_path = retriever.get_image_path(img_id)
            if img_path and Path(img_path).exists():
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center", fontsize=8)
            else:
                ax.text(0.5, 0.5, img_id[:12], ha="center", va="center", fontsize=8)

            is_relevant = img_id in relevant_ids
            border_color = "green" if is_relevant else "red"
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             linewidth=3, edgecolor=border_color, facecolor="none")
            ax.add_patch(rect)

            ax.set_title(f"#{i+1}\n{score:.3f}", fontsize=7)
            ax.axis("off")

        # Hide unused axes
        for i in range(len(results), n_cols):
            axes[i].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        safe_query = query_text[:40].replace(" ", "_").replace("/", "_").replace("'", "")
        fname = f"{version}_{q['id']}_{safe_query}.png"
        plt.savefig(output_dir / fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Visual debugging of retrieval results")
    parser.add_argument("--version", default="v0b", help="Version to visualize")
    parser.add_argument("--index_dir", default="indexes")
    parser.add_argument("--data_dir", default="data/fashionpedia")
    parser.add_argument("--annotations", default="data/fashionpedia/instances_attributes_val2020.json")
    parser.add_argument("--captions", default="indexes/v3/captions.json")
    parser.add_argument("--output_dir", default="evaluation/debug_plots")
    parser.add_argument("--base_only", action="store_true", help="Use 5 base queries only")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    queries = get_base_queries() if args.base_only else get_all_queries()

    # Get retriever
    from evaluation.evaluate import get_retriever
    retriever = get_retriever(args.version, args.index_dir, args.data_dir)

    # GT generator
    gt_gen = GroundTruthGenerator(
        annotations_path=args.annotations,
        captions_path=args.captions if Path(args.captions).exists() else None,
    )

    print(f"\nVisualizing {args.version} on {len(queries)} queries...")
    visualize_query_results(
        version=args.version,
        queries=queries,
        retriever=retriever,
        gt_gen=gt_gen,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
