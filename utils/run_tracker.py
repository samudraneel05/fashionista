"""Run tracker: logs metrics across versions and runs for comparison.

Saves results to JSON, supports loading previous runs for comparison,
and provides a simple API for logging and retrieving metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class RunRecord:
    """Single run record."""
    run_id: str
    version: str
    timestamp: str
    metrics: Dict[str, float]
    per_category: Dict[str, Dict[str, float]]
    config: Dict[str, Any] = None
    notes: str = ""


class RunTracker:
    """Tracks and persists evaluation runs across versions.

    Stores all runs in a single JSON file for easy comparison and visualization.
    """

    def __init__(self, tracker_path: str = "evaluation/run_history.json"):
        self.tracker_path = Path(tracker_path)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: List[RunRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing run history."""
        if self.tracker_path.exists():
            with open(self.tracker_path, "r") as f:
                data = json.load(f)
            self.runs = [RunRecord(**r) for r in data]

    def _save(self) -> None:
        """Save run history to disk."""
        with open(self.tracker_path, "w") as f:
            json.dump([asdict(r) for r in self.runs], f, indent=2)

    def log_run(
        self,
        version: str,
        metrics: Dict[str, float],
        per_category: Dict[str, Dict[str, float]],
        config: Dict[str, Any] = None,
        notes: str = "",
    ) -> RunRecord:
        """Log a new run.

        Args:
            version: Version name (v0a, v0b, v1, etc.)
            metrics: Overall metrics dict (precision_at_5, precision_at_10, etc.)
            per_category: Per-category metrics dict.
            config: Optional config dict (model, batch_size, num_images, etc.)
            notes: Optional notes string.

        Returns:
            The created RunRecord.
        """
        run_id = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = RunRecord(
            run_id=run_id,
            version=version,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            per_category=per_category,
            config=config or {},
            notes=notes,
        )
        self.runs.append(record)
        self._save()
        print(f"Logged run: {run_id}")
        return record

    def get_runs_by_version(self, version: str) -> List[RunRecord]:
        """Get all runs for a specific version."""
        return [r for r in self.runs if r.version == version]

    def get_latest_run(self, version: str) -> Optional[RunRecord]:
        """Get the most recent run for a version."""
        runs = self.get_runs_by_version(version)
        return runs[-1] if runs else None

    def get_all_versions(self) -> List[str]:
        """Get sorted list of all versions that have been run."""
        return sorted(set(r.version for r in self.runs))

    def get_latest_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get latest metrics for each version for comparison.

        Returns:
            {version: {metric_name: value, ...}, ...}
        """
        comparison = {}
        for version in self.get_all_versions():
            latest = self.get_latest_run(version)
            if latest:
                comparison[version] = latest.metrics
        return comparison

    def summary_table(self) -> str:
        """Generate a readable summary table of all runs."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"  Run History ({len(self.runs)} total runs)")
        lines.append(f"{'='*80}")
        lines.append(f"\n  {'Version':<12} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'MRR':>8} {'Latency':>10} {'Timestamp'}")
        lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*25}")

        for version in self.get_all_versions():
            latest = self.get_latest_run(version)
            if latest:
                m = latest.metrics
                lines.append(
                    f"  {version:<12} {m.get('precision_at_5', 0):>8.4f} {m.get('precision_at_10', 0):>8.4f} "
                    f"{m.get('recall_at_10', 0):>8.4f} {m.get('mrr', 0):>8.4f} "
                    f"{m.get('latency_ms', 0):>8.1f}ms {latest.timestamp[:19]}"
                )

        lines.append(f"\n{'='*80}\n")
        return "\n".join(lines)
