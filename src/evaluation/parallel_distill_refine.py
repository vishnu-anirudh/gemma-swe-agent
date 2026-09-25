"""Parallel-Distill-Refine (PDR) and Recursive Tournament Voting (RTV) for Test-Time Scaling.

Implements SOTA test-time compute optimization for sub-3B coding agents:
1. Parallel Exploration: Samples K diverse candidate patches.
2. Trajectory Distillation: Compresses failed attempts and error traces into a compact diagnostic context.
3. Refined Generation: Generates secondary rollouts conditioned on distilled failure lessons.
4. Recursive Tournament Selection: Selects the Pareto-optimal candidate using PRM + verifier rubric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data.sri_formatter import SRIBlock, SRIFormatter
from src.rlvr.prm import RubricProcessRewardModel, PRMScore


@dataclass
class CandidateHypothesis:
    """Represents a single candidate generation and its evaluation metrics."""

    candidate_id: int
    raw_output: str
    sri_blocks: List[SRIBlock] = field(default_factory=list)
    prm_score: float = 0.0
    rubric_detail: Optional[PRMScore] = None
    passed_tests: bool = False
    test_stdout: str = ""
    test_stderr: str = ""
    error_summary: Optional[str] = None
    composite_score: float = 0.0


@dataclass
class PDRResult:
    """Final output of the Parallel-Distill-Refine procedure."""

    winning_candidate: CandidateHypothesis
    all_candidates: List[CandidateHypothesis]
    distilled_feedback: str
    pass_at_k: bool
    iterations_run: int


class ParallelDistillRefine:
    """Orchestrates test-time scaling via Parallel-Distill-Refine and Tournament Voting."""

    def __init__(
        self,
        prm: Optional[RubricProcessRewardModel] = None,
        k: int = 4,
        max_refine_rounds: int = 1,
    ):
        self.prm = prm or RubricProcessRewardModel()
        self.k = k
        self.max_refine_rounds = max_refine_rounds

    def distill_failure_analysis(self, candidates: List[CandidateHypothesis]) -> str:
        """Condense failed candidate rollouts into a structured, compact failure summary."""
        failures = []
        for cand in candidates:
            if cand.passed_tests:
                continue

            reason = "Syntax/Formatting: No valid SRI search-and-replace block found."
            if cand.sri_blocks:
                if cand.test_stderr:
                    # Extract last 2 lines of stderr (e.g. AssertionError or Exception)
                    lines = [ln.strip() for ln in cand.test_stderr.strip().split("\n") if ln.strip()]
                    err_msg = " | ".join(lines[-2:]) if lines else "Non-zero exit code."
                    reason = f"Test Failure: {err_msg}"
                elif cand.error_summary:
                    reason = cand.error_summary
                else:
                    reason = "Test suite failed."

            failures.append(f"• Candidate #{cand.candidate_id}: {reason}")

        if not failures:
            return "All candidate hypotheses passed or no explicit errors detected."

        distilled = (
            "=== Distilled Analysis of Prior Candidate Attempts ===\n"
            "The following failure modes were encountered during initial rollouts:\n"
            + "\n".join(failures[:4])
            + "\n\nCorrection Objective: Avoid the above syntax and regression errors. "
            "Ensure the search block matches the repository file exactly and addresses the root cause."
        )
        return distilled

    def score_candidate(
        self,
        candidate_id: int,
        raw_output: str,
        verify_fn: Optional[Callable[[List[SRIBlock]], Tuple[bool, str, str]]] = None,
    ) -> CandidateHypothesis:
        """Evaluate a single candidate using PRM and optional test verifier."""
        blocks = SRIFormatter.parse(raw_output)
        rubric = self.prm.evaluate_trajectory(raw_output)

        passed = False
        stdout = ""
        stderr = ""
        err_msg = None

        if verify_fn is not None and len(blocks) > 0:
            try:
                passed, stdout, stderr = verify_fn(blocks)
            except Exception as e:
                err_msg = f"Verifier error: {e}"

        # Compute composite tournament score:
        # Verified fix receives heavy boost (100.0)
        # Syntactic validity and PRM rubric score provide fine-grained ranking
        composite = (100.0 if passed else 0.0) + (10.0 if len(blocks) > 0 else -10.0) + rubric.total_score

        return CandidateHypothesis(
            candidate_id=candidate_id,
            raw_output=raw_output,
            sri_blocks=blocks,
            prm_score=rubric.total_score,
            rubric_detail=rubric,
            passed_tests=passed,
            test_stdout=stdout,
            test_stderr=stderr,
            error_summary=err_msg,
            composite_score=composite,
        )

    def recursive_tournament_vote(
        self, candidates: List[CandidateHypothesis]
    ) -> CandidateHypothesis:
        """Execute tournament voting to select the best candidate."""
        if not candidates:
            raise ValueError("No candidates provided for tournament voting.")

        # Sort descending by composite score
        sorted_cands = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
        return sorted_cands[0]

    def run_pdr(
        self,
        initial_prompt: str,
        generate_fn: Callable[[str, int], List[str]],
        verify_fn: Optional[Callable[[List[SRIBlock]], Tuple[bool, str, str]]] = None,
    ) -> PDRResult:
        """Run complete Parallel-Distill-Refine cycle.

        Args:
            initial_prompt: Problem prompt for the agent.
            generate_fn: Function that accepts (prompt, count) and returns list of completions.
            verify_fn: Optional callback that takes List[SRIBlock] and returns (passed, stdout, stderr).
        """
        # Step 1: Parallel Generation (K candidates)
        raw_candidates = generate_fn(initial_prompt, self.k)
        candidates: List[CandidateHypothesis] = [
            self.score_candidate(idx + 1, text, verify_fn)
            for idx, text in enumerate(raw_candidates)
        ]

        any_passed = any(c.passed_tests for c in candidates)
        distilled_text = ""

        # Step 2: Distill and Refine if no candidate passed and refinement rounds permitted
        if not any_passed and self.max_refine_rounds > 0 and verify_fn is not None:
            distilled_text = self.distill_failure_analysis(candidates)
            refine_prompt = (
                f"{initial_prompt}\n\n"
                f"{distilled_text}\n\n"
                f"Please provide your refined, verified fix in Search-and-Replace Infilling (SRI) format:"
            )

            # Generate refined rollouts (half-K candidates for refinement)
            refine_k = max(2, self.k // 2)
            refined_raw = generate_fn(refine_prompt, refine_k)
            for r_idx, r_text in enumerate(refined_raw):
                refined_cand = self.score_candidate(self.k + r_idx + 1, r_text, verify_fn)
                candidates.append(refined_cand)

            any_passed = any(c.passed_tests for c in candidates)

        # Step 3: Recursive Tournament Selection
        winner = self.recursive_tournament_vote(candidates)

        return PDRResult(
            winning_candidate=winner,
            all_candidates=candidates,
            distilled_feedback=distilled_text,
            pass_at_k=any_passed,
            iterations_run=1 + (1 if distilled_text else 0),
        )
