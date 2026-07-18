"""Visualization module: generates charts comparing versions and ablations.

Produces:
- Bar charts comparing P@5, P@10, R@10, MRR across versions
- Per-category heatmap showing which version excels where
- Latency comparison
- Ablation impact chart
- Saves all plots as PNG files for the writeup
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional


# Color palette for versions
VERSION_COLORS = {
    "v0a": "#e74c3c",  # red
    "v0b": "#e67e22",  # orange
    "v1": "#f1c40f",   # yellow
    "v2": "#2ecc71",   # green
    "v3": "#1abc9c",   # teal
    "v4": "#3498db",   # blue
}

METRIC_LABELS = {
    "precision_at_5": "Precision@5",
    "precision_at_10": "Precision@10",
    "recall_at_10": "Recall@10",
    "mrr": "MRR",
    "latency_ms": "Latency (ms)",
}

CATEGORY_ORDER = ["attribute", "contextual", "complex", "style", "compositional"]
CATEGORY_LABELS = {
    "attribute": "Attribute\nSpecific",
    "contextual": "Contextual\n/Place",
    "complex": "Complex\nSemantic",
    "style": "Style\nInference",
    "compositional": "Compositional",
}


def plot_metric_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "precision_at_5",
    output_path: str = "evaluation/plots/metric_comparison.png",
    title: str = None,
) -> str:
    """Bar chart comparing a single metric across all versions.

    Args:
        results: {version: {"aggregated": {"overall": {...}, "per_category": {...}}}}
        metric: Metric to plot.
        output_path: Where to save the PNG.
        title: Custom chart title.

    Returns:
        Path to saved PNG.
    """
    versions = sorted(results.keys())
    values = [results[v]["aggregated"]["overall"].get(metric, 0) for v in versions]
    colors = [VERSION_COLORS.get(v, "#95a5a6") for v in versions]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(versions, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Version", fontsize=12)
    ax.set_title(title or f"{metric_label} Across Versions", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_all_metrics(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "evaluation/plots",
) -> List[str]:
    """Generate bar charts for all metrics.

    Returns:
        List of saved PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for metric in ["precision_at_5", "precision_at_10", "recall_at_10", "mrr"]:
        path = str(output_dir / f"{metric}.png")
        paths.append(plot_metric_comparison(results, metric, path))

    return paths


def plot_per_category_heatmap(
    results: Dict[str, Dict[str, Any]],
    metric: str = "precision_at_5",
    output_path: str = "evaluation/plots/per_category_heatmap.png",
) -> str:
    """Heatmap showing per-category performance across versions.

    Rows = versions, Columns = query categories.
    """
    versions = sorted(results.keys())
    categories = [c for c in CATEGORY_ORDER if c in
                  set().union(*[r.get("aggregated", {}).get("per_category", {}).keys()
                                  for r in results.values()])]

    if not categories:
        categories = CATEGORY_ORDER

    matrix = np.zeros((len(versions), len(categories)))
    for i, version in enumerate(versions):
        per_cat = results[version].get("aggregated", {}).get("per_category", {})
        for j, cat in enumerate(categories):
            matrix[i, j] = per_cat.get(cat, {}).get(metric, 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(matrix.max(), 0.1))

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories], fontsize=10)
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions, fontsize=11, fontweight="bold")

    # Add value annotations
    for i in range(len(versions)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_title(f"{metric_label} by Query Category", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_latency_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "evaluation/plots/latency.png",
) -> str:
    """Bar chart comparing latency across versions."""
    versions = sorted(results.keys())
    latencies = [results[v]["aggregated"]["overall"].get("latency_ms", 0) for v in versions]
    colors = [VERSION_COLORS.get(v, "#95a5a6") for v in versions]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(versions, latencies, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xlabel("Version", fontsize=12)
    ax.set_title("Retrieval Latency Across Versions", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(latencies) * 1.15 if latencies else 10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_ablation_impact(
    ablation_results: Dict[str, Dict[str, Any]],
    metric: str = "mrr",
    output_path: str = "evaluation/plots/ablation_impact.png",
) -> str:
    """Bar chart showing impact of removing each component from V4.

    Args:
        ablation_results: {ablation_name: {"aggregated": {"overall": {...}}}}
        metric: Metric to plot.
    """
    ablations = sorted(ablation_results.keys())
    values = [ablation_results[a]["aggregated"]["overall"].get(metric, 0) for a in ablations]

    # Highlight full V4
    colors = ["#3498db" if "full" in a else "#e74c3c" for a in ablations]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(ablations, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2.,
                f"{val:.4f}", ha="left", va="center", fontsize=10, fontweight="bold")

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_title(f"V4 Ablation Impact on {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def generate_all_plots(
    results: Dict[str, Dict[str, Any]],
    ablation_results: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: str = "evaluation/plots",
) -> List[str]:
    """Generate all comparison plots.

    Args:
        results: Version evaluation results.
        ablation_results: Optional ablation results.
        output_dir: Directory to save plots.

    Returns:
        List of saved PNG paths.
    """
    paths = []
    paths.extend(plot_all_metrics(results, output_dir))
    paths.append(plot_per_category_heatmap(results, output_path=str(Path(output_dir) / "per_category_heatmap.png")))
    paths.append(plot_latency_comparison(results, output_path=str(Path(output_dir) / "latency.png")))

    if ablation_results:
        paths.append(plot_ablation_impact(ablation_results, "mrr",
                                           output_path=str(Path(output_dir) / "ablation_mrr.png")))
        paths.append(plot_ablation_impact(ablation_results, "precision_at_5",
                                           output_path=str(Path(output_dir) / "ablation_p5.png")))

    print(f"\nGenerated {len(paths)} plots in {output_dir}/")
    return paths


def plot_from_results_file(
    results_path: str = "evaluation/results.json",
    ablation_path: str = "evaluation/ablation_results.json",
    output_dir: str = "evaluation/plots",
) -> List[str]:
    """Load results from JSON files and generate all plots."""
    results = {}
    if Path(results_path).exists():
        with open(results_path, "r") as f:
            results = json.load(f)

    ablation_results = None
    if Path(ablation_path).exists():
        with open(ablation_path, "r") as f:
            ablation_results = json.load(f)

    if not results:
        print("No results found. Run evaluation first.")
        return []

    return generate_all_plots(results, ablation_results, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results", default="evaluation/results.json")
    parser.add_argument("--ablations", default="evaluation/ablation_results.json")
    parser.add_argument("--output_dir", default="evaluation/plots")
    args = parser.parse_args()
    plot_from_results_file(args.results, args.ablations, args.output_dir)
