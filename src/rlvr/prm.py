"""Rubric-Based Process Reward Model (PRM).

Evaluates trajectory intermediate quality against structured rubrics:
- Fault localization
- Trajectory discipline (avoiding repetitive commands/loops)
- SRI patch validity
- System invariant preservation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import re

from src.data.sri_formatter import SRIFormatter


@dataclass
class RubricCriteria:
    """Weighting for different intermediate process dimensions."""

    weight_localization: float = 0.35
    weight_discipline: float = 0.25
    weight_formatting: float = 0.40


@dataclass
class PRMScore:
    """Granular process evaluation score."""

    total_score: float
    localization_score: float
    discipline_score: float
    formatting_score: float
    details: Dict[str, str]


class RubricProcessRewardModel:
    """Evaluates agent rollouts using process rubrics to guide RLVR."""

    def __init__(self, criteria: Optional[RubricCriteria] = None):
        self.criteria = criteria or RubricCriteria()

    @staticmethod
    def extract_observable_files_from_context(context_text: str) -> Set[str]:
        """Extract candidate repository file paths strictly from observable prompt/tool context.

        Parses `=== File: <path> ===` headers, `view_file` tool calls, and `.py` paths
        mentioned in the public issue report or reproduction traceback. Never accesses
        gold solution patches.
        """
        candidates: Set[str] = set()
        for m in re.finditer(r"===\s*(?:Auto-Inspected\s+)?File:\s*([^\s=]+\.py)\s*===", context_text):
            candidates.add(m.group(1).strip().lstrip("./"))
        for m in re.finditer(r'path=["\']([^"\']+\.py)["\']', context_text):
            candidates.add(m.group(1).strip().lstrip("./"))
        for m in re.finditer(r"\b([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)+\.py)\b", context_text):
            candidates.add(m.group(1).strip().lstrip("./"))
        return candidates

    def evaluate_trajectory(
        self,
        trajectory_text: str,
        target_files: Optional[Set[str]] = None,
        max_repeated_commands: int = 2,
        observable_context: Optional[str] = None,
    ) -> PRMScore:
        """Score an agent trajectory rollout against the process rubric.

        If `target_files` is None and `observable_context` is provided, candidate file
        paths are extracted strictly from `observable_context` (issue description,
        reproduction traceback, and `view_file` observations), ensuring zero privileged
        information from gold benchmark patches.
        """
        details: Dict[str, str] = {}

        # 1. Formatting & SRI Validity Score
        # Check if the output contains parseable SRI blocks
        try:
            blocks = SRIFormatter.parse(trajectory_text)
        except Exception:
            blocks = []
        if len(blocks) > 0:
            # Check if all blocks specify existing-looking relative paths
            valid_paths = [b.is_valid() and not b.file_path.startswith("/") for b in blocks]
            formatting_score = 1.0 if all(valid_paths) else 0.5
            details["formatting"] = f"Valid SRI blocks found: {len(blocks)}"
        else:
            # Check if unified diff format was attempted
            if "diff --git" in trajectory_text or "--- a/" in trajectory_text:
                formatting_score = 0.3  # Partial credit, but penalized for not using SRI
                details["formatting"] = "Diff used instead of preferred SRI format"
            else:
                formatting_score = 0.0
                details["formatting"] = "No valid code edit blocks identified"

        # 2. Target Localization Score (Strictly from observable context if target_files not given)
        effective_targets = target_files
        if effective_targets is None and observable_context:
            effective_targets = self.extract_observable_files_from_context(observable_context)

        if effective_targets:
            mentioned = sum(
                1 for target in effective_targets if target.lower() in trajectory_text.lower()
            )
            localization_score = min(1.0, mentioned / max(1, len(effective_targets)))
            details["localization"] = f"Mentioned {mentioned}/{len(effective_targets)} observable context files"
        else:
            # If no observable file hints exist, check if specific code files were targeted in blocks
            localization_score = 1.0 if len(blocks) > 0 else 0.5
            details["localization"] = "Target files evaluated by presence of file targets"

        # 3. Trajectory Discipline Score (Repeated loops & commands)
        command_pattern = re.compile(r"<command>(.*?)</command>", re.DOTALL)
        commands = command_pattern.findall(trajectory_text)
        if commands:
            cmd_counts: Dict[str, int] = {}
            repetitive = False
            for cmd in commands:
                cleaned = cmd.strip()
                cmd_counts[cleaned] = cmd_counts.get(cleaned, 0) + 1
                if cmd_counts[cleaned] > max_repeated_commands:
                    repetitive = True
                    break

            discipline_score = 0.2 if repetitive else 1.0
            details["discipline"] = (
                "Repetitive commands detected" if repetitive else "Disciplined exploration"
            )
        else:
            discipline_score = 1.0
            details["discipline"] = "Single-turn patch (no command loop)"

        # Weighted combination
        total = (
            self.criteria.weight_localization * localization_score
            + self.criteria.weight_discipline * discipline_score
            + self.criteria.weight_formatting * formatting_score
        )

        return PRMScore(
            total_score=round(total, 4),
            localization_score=localization_score,
            discipline_score=discipline_score,
            formatting_score=formatting_score,
            details=details,
        )
