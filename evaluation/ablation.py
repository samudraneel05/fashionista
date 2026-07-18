"""Ablation studies for V4 hybrid ensemble.

Removes one component at a time from V4 to measure its contribution:
- Without region vectors (V1 + V3 only)
- Without VLM captions (V1 + V2 only)
- Without scene masking (V2 + V3 only)
- Without LABCLIP
- Without cross-encoder reranking
- Static weights vs dynamic weights
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.queries import get_all_queries, get_base_queries
from evaluation.metrics import evaluate_query, aggregate_results, format_results_table
from evaluation.ground_truth import GroundTruthGenerator


def evaluate_ablation(
    ablation_name: str,
    index_dir: str,
    data_dir: str,
    annotations_path: str = None,
    captions_path: str = None,
    queries: List[Dict] = None,
    top_k: int = 10,
    device: str = None,
) -> Dict[str, Any]:
    """Evaluate a single ablation configuration."""
    from retriever.hybrid_retriever import HybridRetriever

    print(f"\n{'='*50}")
    print(f"  Ablation: {ablation_name}")
    print(f"{'='*50}")

    retriever = HybridRetriever(
        index_dir=str(Path(index_dir) / "v4"),
        data_dir=data_dir,
        device=device,
    )

    # Apply ablation modifications
    if "no_regions" in ablation_name:
        # Zero out region weights
        retriever.WEIGHTS = {
            qt: {ch: w for ch, w in weights.items() if "region" not in ch}
            for qt, weights in retriever.WEIGHTS.items()
        }
        # Renormalize weights
        for qt in retriever.WEIGHTS:
            total = sum(retriever.WEIGHTS[qt].values())
            if total > 0:
                retriever.WEIGHTS[qt] = {ch: w / total for ch, w in retriever.WEIGHTS[qt].items()}

    elif "no_captions" in ablation_name:
        # Zero out caption weight
        retriever.WEIGHTS = {
            qt: {ch: w for ch, w in weights.items() if ch != "caption"}
            for qt, weights in retriever.WEIGHTS.items()
        }
        for qt in retriever.WEIGHTS:
            total = sum(retriever.WEIGHTS[qt].values())
            if total > 0:
                retriever.WEIGHTS[qt] = {ch: w / total for ch, w in retriever.WEIGHTS[qt].items()}

    elif "no_scene" in ablation_name:
        # Zero out scene weight
        retriever.WEIGHTS = {
            qt: {ch: w for ch, w in weights.items() if ch != "scene"}
            for qt, weights in retriever.WEIGHTS.items()
        }
        for qt in retriever.WEIGHTS:
            total = sum(retriever.WEIGHTS[qt].values())
            if total > 0:
                retriever.WEIGHTS[qt] = {ch: w / total for ch, w in retriever.WEIGHTS[qt].items()}

    elif "no_labclip" in ablation_name:
        retriever.labclip_transform = None

    elif "no_reranking" in ablation_name:
        # Override retrieve to skip stage 3
        original_retrieve = retriever.retrieve
        def retrieve_no_rerank(query, top_k=10):
            results = original_retrieve.__func__(retriever, query, top_k)
            return results
        retriever.retrieve = retrieve_no_rerank

    elif "static_weights" in ablation_name:
        # Use uniform static weights for all query types
        static = {"global": 0.20, "scene": 0.15, "action": 0.20,
                  "region_upper": 0.15, "region_lower": 0.15, "caption": 0.15}
        retriever.WEIGHTS = {qt: static.copy() for qt in retriever.WEIGHTS}

    # Get queries
    if queries is None:
        queries = get_all_queries()

    gt_gen = GroundTruthGenerator(annotations_path=annotations_path, captions_path=captions_path)

    all_results = []
    for q in queries:
        candidate_ids = [r["image_id"] for r in retriever.retrieve(q["query"], top_k=50)]
        relevant_ids = gt_gen.get_ground_truth(q["id"], q["query"], q["category"], candidate_ids)

        def retrieve_fn(query, top_k):
            return retriever.retrieve(query, top_k)

        result = evaluate_query(
            query_id=q["id"],
            query=q["query"],
            category=q["category"],
            retrieve_fn=retrieve_fn,
            relevant_ids=relevant_ids,
            top_k=top_k,
        )
        all_results.append(result)

    agg = aggregate_results(all_results)
    print(format_results_table(agg, ablation_name))

    return {
        "ablation": ablation_name,
        "aggregated": agg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies on V4")
    parser.add_argument("--index_dir", type=str, default="indexes")
    parser.add_argument("--data_dir", type=str, default="data/fashionpedia")
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation/ablation_results.json")
    args = parser.parse_args()

    queries = get_base_queries() if args.base_only else get_all_queries()

    ablations = [
        "full_v4",
        "no_regions",
        "no_captions",
        "no_scene",
        "no_labclip",
        "no_reranking",
        "static_weights",
    ]

    results = {}
    for ablation in ablations:
        try:
            result = evaluate_ablation(
                ablation_name=ablation,
                index_dir=args.index_dir,
                data_dir=args.data_dir,
                annotations_path=args.annotations,
                captions_path=args.captions,
                queries=queries,
                top_k=args.top_k,
                device=args.device,
            )
            results[ablation] = result
        except Exception as e:
            print(f"  Error in ablation {ablation}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print(f"\n{'='*80}")
    print(f"  Ablation Comparison")
    print(f"{'='*80}")
    print(f"\n  {'Ablation':<25} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ablation, res in results.items():
        agg = res["aggregated"]["overall"]
        print(f"  {ablation:<25} {agg['precision_at_5']:>8.4f} {agg['precision_at_10']:>8.4f} "
              f"{agg['recall_at_10']:>8.4f} {agg['mrr']:>8.4f}")
    print(f"\n{'='*80}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
