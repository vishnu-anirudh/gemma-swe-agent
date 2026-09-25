"""Step-Level Error Masking for Trajectory Supervised Fine-Tuning (SFT).

Implements step-level masking to exclude tokens corresponding to intermediate
execution-time errors and failed test validations from the training loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class TrajectoryStep:
    """A single action-observation step in an agent exploration trajectory."""

    action_text: str
    observation_text: str
    is_error: bool = False
    is_tool_call: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class StepLevelErrorMasker:
    """Constructs token sequences and target labels with step-level error masking (-100)."""

    def __init__(
        self,
        tokenizer: Any,
        ignore_index: int = -100,
        mask_observations: bool = True,
        mask_error_actions: bool = True,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.mask_observations = mask_observations
        self.mask_error_actions = mask_error_actions

    def format_and_mask_trajectory(
        self,
        system_prompt: str,
        steps: List[TrajectoryStep],
        final_solution: Optional[str] = None,
        max_length: int = 4096,
    ) -> Dict[str, torch.Tensor]:
        """Convert a sequence of trajectory steps into input_ids and labels with masked loss.

        Tokens corresponding to:
        1. System prompts (prompt tokens)
        2. Observations (environmental feedback)
        3. Erroneous actions that led to immediate syntax/tool execution failures
        are set to `ignore_index` (-100) so cross-entropy loss is computed ONLY on
        constructive, valid actions and the final correct patch.
        """
        all_token_ids: List[int] = []
        all_labels: List[int] = []

        # 1. System prompt (never compute loss on system prompt)
        if system_prompt:
            sys_ids = self.tokenizer.encode(system_prompt, add_special_tokens=False)
            all_token_ids.extend(sys_ids)
            all_labels.extend([self.ignore_index] * len(sys_ids))

        # 2. Iterate through steps
        for step in steps:
            # Action tokens
            action_ids = self.tokenizer.encode(
                f"\n<action>\n{step.action_text}\n</action>",
                add_special_tokens=False,
            )
            all_token_ids.extend(action_ids)

            # If this action was an intermediate error or syntax bug, mask its loss
            if step.is_error and self.mask_error_actions:
                all_labels.extend([self.ignore_index] * len(action_ids))
            else:
                all_labels.extend(action_ids)

            # Observation tokens (environment feedback - always masked from target loss)
            obs_ids = self.tokenizer.encode(
                f"\n<observation>\n{step.observation_text}\n</observation>",
                add_special_tokens=False,
            )
            all_token_ids.extend(obs_ids)
            if self.mask_observations:
                all_labels.extend([self.ignore_index] * len(obs_ids))
            else:
                all_labels.extend(obs_ids)

        # 3. Final solution tokens (always trained on)
        if final_solution:
            solution_ids = self.tokenizer.encode(
                f"\n<final_patch>\n{final_solution}\n</final_patch>",
                add_special_tokens=False,
            )
            all_token_ids.extend(solution_ids)
            all_labels.extend(solution_ids)

        # Truncate if exceeding max_length
        if len(all_token_ids) > max_length:
            all_token_ids = all_token_ids[:max_length]
            all_labels = all_labels[:max_length]

        input_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
        }
