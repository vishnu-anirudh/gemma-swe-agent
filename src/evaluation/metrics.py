"""Evaluation metrics for SWE-bench benchmarks.

Computes Pass@1, Pass@K, patch application rates, and formats rich
leaderboard-style summary reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table


@dataclass
class InstanceEvaluationResult:
    """Evaluation result for an individual benchmark instance."""

    instance_id: str
    resolved: bool
    patch_applied: bool
    format_type: str
    fail_to_pass_passed: bool
    pass_to_pass_passed: bool
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark metrics across an evaluation dataset."""

    total_instances: int
    resolved_instances: int
    unresolved_instances: int
    pass_at_1: float
    patch_application_rate: float
    results: List[InstanceEvaluationResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """Render a formatted summary table to console."""
        console = Console()
        table = Table(title="SWE-bench Evaluation Summary", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Total Evaluated Instances", str(self.total_instances))
        table.add_row("Resolved Instances", str(self.resolved_instances))
        table.add_row("Unresolved Instances", str(self.unresolved_instances))
        table.add_row("Pass@1 Resolution Rate", f"{self.pass_at_1:.2%}")
        table.add_row("Patch Application Rate", f"{self.patch_application_rate:.2%}")

        console.print(table)


class MetricsCalculator:
    """Computes benchmark metrics including Pass@1 and Pass@K."""

    @classmethod
    def compute_summary(
        cls, results: List[InstanceEvaluationResult]
    ) -> BenchmarkSummary:
        """Compute aggregated Pass@1 and application rate from instance results."""
        total = len(results)
        if total == 0:
            return BenchmarkSummary(
                total_instances=0,
                resolved_instances=0,
                unresolved_instances=0,
                pass_at_1=0.0,
                patch_application_rate=0.0,
                results=[],
            )

        resolved = sum(1 for r in results if r.resolved)
        applied = sum(1 for r in results if r.patch_applied)

        return BenchmarkSummary(
            total_instances=total,
            resolved_instances=resolved,
            unresolved_instances=total - resolved,
            pass_at_1=resolved / total,
            patch_application_rate=applied / total,
            results=results,
        )

    @classmethod
    def compute_pass_at_k(cls, n: int, c: int, k: int) -> float:
        """Compute unbiased Pass@K metric (Chen et al., 2021).

        Args:
            n: Total number of sampled rollouts per problem.
            c: Number of correct/resolved rollouts.
            k: K-value (e.g. 1, 5, 10).
        """
        if n - c < k:
            return 1.0
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))
