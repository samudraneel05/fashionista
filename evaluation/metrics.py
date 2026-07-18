"""Evaluation metrics: Precision@K, Recall@K, MRR, Latency."""

import time
import numpy as np
from typing import List, Dict, Any, Set, Callable
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single query evaluation result."""
    query_id: str
    query: str
    category: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    latency_ms: float


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K: fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for img_id in top_k if img_id in relevant_ids)
    return relevant_count / k


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K: fraction of all relevant items found in top-k."""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for img_id in top_k if img_id in relevant_ids)
    return found / len(relevant_ids)


def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, img_id in enumerate(retrieved_ids):
        if img_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_query(
    query_id: str,
    query: str,
    category: str,
    retrieve_fn: Callable,
    relevant_ids: Set[str],
    top_k: int = 10,
) -> RetrievalResult:
    """Evaluate a single query.

    Args:
        query_id: Unique query identifier.
        query: Natural language query string.
        category: Query category (attribute, contextual, complex, style, compositional).
        retrieve_fn: Function that takes (query, top_k) and returns list of dicts with image_id.
        relevant_ids: Set of ground-truth relevant image IDs.
        top_k: Number of results to retrieve.

    Returns:
        RetrievalResult with metrics.
    """
    start_time = time.time()
    results = retrieve_fn(query, top_k)
    latency_ms = (time.time() - start_time) * 1000

    retrieved_ids = [r["image_id"] for r in results]

    p5 = compute_precision_at_k(retrieved_ids, relevant_ids, 5)
    p10 = compute_precision_at_k(retrieved_ids, relevant_ids, 10)
    r10 = compute_recall_at_k(retrieved_ids, relevant_ids, 10)
    mrr = compute_mrr(retrieved_ids, relevant_ids)

    return RetrievalResult(
        query_id=query_id,
        query=query,
        category=category,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        precision_at_5=p5,
        precision_at_10=p10,
        recall_at_10=r10,
        mrr=mrr,
        latency_ms=latency_ms,
    )


def aggregate_results(results: List[RetrievalResult]) -> Dict[str, Any]:
    """Aggregate evaluation results across all queries.

    Returns:
        Dict with overall metrics and per-category breakdowns.
    """
    if not results:
        return {}

    overall = {
        "precision_at_5": np.mean([r.precision_at_5 for r in results]),
        "precision_at_10": np.mean([r.precision_at_10 for r in results]),
        "recall_at_10": np.mean([r.recall_at_10 for r in results]),
        "mrr": np.mean([r.mrr for r in results]),
        "latency_ms": np.median([r.latency_ms for r in results]),
        "num_queries": len(results),
    }

    # Per-category breakdown
    categories = set(r.category for r in results)
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        per_category[cat] = {
            "precision_at_5": np.mean([r.precision_at_5 for r in cat_results]),
            "precision_at_10": np.mean([r.precision_at_10 for r in cat_results]),
            "recall_at_10": np.mean([r.recall_at_10 for r in cat_results]),
            "mrr": np.mean([r.mrr for r in cat_results]),
            "latency_ms": np.median([r.latency_ms for r in cat_results]),
            "num_queries": len(cat_results),
        }

    return {"overall": overall, "per_category": per_category}


def format_results_table(agg: Dict[str, Any], version_name: str) -> str:
    """Format aggregated results as a readable table."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Results: {version_name}")
    lines.append(f"{'='*70}")
    lines.append(f"\n  Overall (n={agg['overall']['num_queries']}):")
    lines.append(f"    P@5:      {agg['overall']['precision_at_5']:.4f}")
    lines.append(f"    P@10:     {agg['overall']['precision_at_10']:.4f}")
    lines.append(f"    R@10:     {agg['overall']['recall_at_10']:.4f}")
    lines.append(f"    MRR:      {agg['overall']['mrr']:.4f}")
    lines.append(f"    Latency:  {agg['overall']['latency_ms']:.1f}ms")

    lines.append(f"\n  Per-Category:")
    lines.append(f"    {'Category':<20} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8} {'Latency':>10}")
    lines.append(f"    {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for cat, metrics in sorted(agg["per_category"].items()):
        lines.append(
            f"    {cat:<20} {metrics['precision_at_5']:>8.4f} {metrics['precision_at_10']:>8.4f} "
            f"{metrics['recall_at_10']:>8.4f} {metrics['mrr']:>8.4f} {metrics['latency_ms']:>8.1f}ms"
        )
    lines.append(f"\n{'='*70}\n")
    return "\n".join(lines)
