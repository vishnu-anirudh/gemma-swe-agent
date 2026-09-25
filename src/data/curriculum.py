"""Curriculum Learning Dataset Loader for Multi-Turn Trajectory SFT.

Categorizes software engineering trajectories by difficulty (interaction turns)
and applies Step-Level Error Masking for progressive curriculum training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
import torch
from torch.utils.data import Dataset

from src.data.masking import StepLevelErrorMasker, TrajectoryStep


@dataclass
class CuratedTrajectory:
    """Represents a complete multi-turn software engineering trajectory."""

    instance_id: str
    difficulty: str  # 'easy', 'medium', 'hard'
    system_prompt: str
    steps: List[TrajectoryStep]
    final_patch: str


class CurriculumTrajectoryDataset(Dataset):
    """Dataset providing trajectories organized by difficulty curriculum."""

    def __init__(
        self,
        trajectories: List[CuratedTrajectory],
        masker: StepLevelErrorMasker,
        active_difficulty: Optional[str] = None,
        max_length: int = 2048,
    ):
        self.masker = masker
        self.active_difficulty = active_difficulty
        self.max_length = max_length

        if active_difficulty:
            self.trajectories = [t for t in trajectories if t.difficulty == active_difficulty]
        else:
            # Order by curriculum: Easy -> Medium -> Hard
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            self.trajectories = sorted(
                trajectories, key=lambda t: diff_order.get(t.difficulty.lower(), 1)
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.trajectories[idx]
        return self.masker.format_and_mask_trajectory(
            system_prompt=item.system_prompt,
            steps=item.steps,
            final_solution=item.final_patch,
            max_length=self.max_length,
        )

    @classmethod
    def create_synthetic_curriculum(cls, masker: StepLevelErrorMasker) -> CurriculumTrajectoryDataset:
        """Create a synthetic curriculum dataset demonstrating multi-turn trajectories across difficulty levels."""
        from src.data.sri_formatter import SRIFormatter

        trajectories = [
            # Easy: 1 turn, localized single-file fix
            CuratedTrajectory(
                instance_id="easy_arithmetic_fix",
                difficulty="easy",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"calc.py\" start_line=\"1\" end_line=\"10\" />",
                        observation_text="def add(a, b):\n    return a - b",
                        is_error=False,
                    )
                ],
                final_patch=SRIFormatter.format_block("calc.py", "    return a - b", "    return a + b"),
            ),
            # Medium: 3 turns, error recovery and test execution
            CuratedTrajectory(
                instance_id="medium_connection_pool_fix",
                difficulty="medium",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"run_test\" command=\"pytest tests/test_pool.py\" />",
                        observation_text="FAILED tests/test_pool.py::test_reconnect - TimeoutError: connection dropped",
                        is_error=True,  # Execution error: masked!
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"pool.py\" start_line=\"40\" end_line=\"60\" />",
                        observation_text="45 | def reconnect(self):\n46 |     if not self.closed:\n47 |         pass",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"search_code\" query=\"self.closed\" />",
                        observation_text="pool.py:46:    if not self.closed:\npool.py:80:        self.closed = True",
                        is_error=False,
                    ),
                ],
                final_patch=SRIFormatter.format_block(
                    "pool.py",
                    "    if not self.closed:\n        pass",
                    "    if not self.closed:\n        self.connect()",
                ),
            ),
            # Hard: Multi-step cross-module refactor
            CuratedTrajectory(
                instance_id="hard_django_query_fix",
                difficulty="hard",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"search_code\" query=\"FilteredRelation\" />",
                        observation_text="django/db/models/sql/query.py:120: class FilteredRelation",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"django/db/models/sql/query.py\" start_line=\"120\" end_line=\"150\" />",
                        observation_text="125 | def resolve_expression(self):\n126 |     return self.relation_name",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"run_test\" command=\"pytest tests/queries/test_filtered_relation.py\" />",
                        observation_text="FAILED tests/queries/test_filtered_relation.py::test_aggregate - AttributeError: 'NoneType' object has no attribute 'name'",
                        is_error=True,  # Masked!
                    ),
                ],
                final_patch=SRIFormatter.format_block(
                    "django/db/models/sql/query.py",
                    "    return self.relation_name",
                    "    return self.relation_name if self.relation_name else None",
                ),
            ),
        ]

        return cls(trajectories=trajectories, masker=masker)
