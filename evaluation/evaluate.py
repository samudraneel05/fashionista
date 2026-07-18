"""Main evaluation script — runs all queries across all versions and reports metrics."""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

from evaluation.queries import get_all_queries, get_base_queries
from evaluation.metrics import evaluate_query, aggregate_results, format_results_table
from evaluation.ground_truth import GroundTruthGenerator
from utils.run_tracker import RunTracker


def get_retriever(version: str, index_dir: str, data_dir: str, device: str = None):
    """Get the appropriate retriever for a version."""
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


def evaluate_version(
    version: str,
    index_dir: str,
    data_dir: str,
    annotations_path: str = None,
    captions_path: str = None,
    queries: List[Dict] = None,
    top_k: int = 10,
    device: str = None,
) -> Dict[str, Any]:
    """Evaluate a single version on all queries.

    Args:
        version: Version name (v0a, v0b, v1, v2, v3, v4).
        index_dir: Directory containing version subdirectories with FAISS indexes.
        data_dir: Directory containing image data.
        annotations_path: Path to Fashionpedia annotations JSON.
        captions_path: Path to VLM captions JSON.
        queries: List of query dicts. If None, uses all 100 queries.
        top_k: Number of results to retrieve per query.
        device: Device for model inference.

    Returns:
        Dict with aggregated results and per-query details.
    """
    print(f"\n{'='*50}")
    print(f"  Evaluating: {version}")
    print(f"{'='*50}")

    # Get retriever
    retriever = get_retriever(version, index_dir, data_dir, device)

    # Get queries
    if queries is None:
        queries = get_all_queries()

    # Ground truth generator
    gt_gen = GroundTruthGenerator(annotations_path=annotations_path, captions_path=captions_path)

    # Evaluate each query
    all_results = []
    for q in queries:
        # Get ground truth
        candidate_ids = [r["image_id"] for r in retriever.retrieve(q["query"], top_k=50)]
        relevant_ids = gt_gen.get_ground_truth(q["id"], q["query"], q["category"], candidate_ids)

        # Evaluate
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

    # Aggregate
    agg = aggregate_results(all_results)

    # Print results
    print(format_results_table(agg, version))

    return {
        "version": version,
        "aggregated": agg,
        "per_query": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "category": r.category,
                "precision_at_5": r.precision_at_5,
                "precision_at_10": r.precision_at_10,
                "recall_at_10": r.recall_at_10,
                "mrr": r.mrr,
                "latency_ms": r.latency_ms,
                "retrieved_ids": r.retrieved_ids,
                "relevant_ids": list(r.relevant_ids),
            }
            for r in all_results
        ],
    }


def compare_versions(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate comparison table across all versions."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  Version Comparison")
    lines.append(f"{'='*80}")
    lines.append(f"\n  {'Version':<10} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8} {'Latency':>10}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for version, res in sorted(results.items()):
        agg = res["aggregated"]["overall"]
        lines.append(
            f"  {version:<10} {agg['precision_at_5']:>8.4f} {agg['precision_at_10']:>8.4f} "
            f"{agg['recall_at_10']:>8.4f} {agg['mrr']:>8.4f} {agg['latency_ms']:>8.1f}ms"
        )

    lines.append(f"\n{'='*80}\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fashion retrieval versions")
    parser.add_argument("--index_dir", type=str, default="indexes", help="Directory with version indexes")
    parser.add_argument("--data_dir", type=str, default="data/fashionpedia", help="Image data directory")
    parser.add_argument("--annotations", type=str, default=None, help="Fashionpedia annotations JSON path")
    parser.add_argument("--captions", type=str, default=None, help="VLM captions JSON path")
    parser.add_argument("--versions", nargs="+", default=["v0a", "v0b"], help="Versions to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all versions")
    parser.add_argument("--base_only", action="store_true", help="Use only 5 base queries")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--output", type=str, default="evaluation/results.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.all:
        args.versions = ["v0a", "v0b", "v1", "v2", "v3", "v4"]

    queries = get_base_queries() if args.base_only else get_all_queries()

    all_results = {}
    for version in args.versions:
        try:
            result = evaluate_version(
                version=version,
                index_dir=args.index_dir,
                data_dir=args.data_dir,
                annotations_path=args.annotations,
                captions_path=args.captions,
                queries=queries,
                top_k=args.top_k,
                device=args.device,
            )
            all_results[version] = result
        except Exception as e:
            print(f"  Error evaluating {version}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print(compare_versions(all_results))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Log runs to tracker
    tracker = RunTracker()
    for version, result in all_results.items():
        agg = result["aggregated"]
        tracker.log_run(
            version=version,
            metrics=agg["overall"],
            per_category=agg["per_category"],
            config={"top_k": args.top_k, "num_queries": len(queries),
                    "base_only": args.base_only},
        )
    print(tracker.summary_table())

    # Generate plots
    try:
        from evaluation.visualize import generate_all_plots
        ablation_results = None
        ablation_path = Path("evaluation/ablation_results.json")
        if ablation_path.exists():
            with open(ablation_path, "r") as f:
                ablation_results = json.load(f)
        generate_all_plots(all_results, ablation_results)
    except Exception as e:
        print(f"Plot generation skipped: {e}")


if __name__ == "__main__":
    main()
