"""Adaptive Virtual Sample Policy Optimization (AVSPO).

Detects and neutralizes Advantage Collapse in GRPO when evaluating sparse,
binary software unit-test rewards (all pass or all fail).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class AVSPOStats:
    """Statistics for advantage collapse monitoring."""

    total_groups: int = 0
    collapsed_groups: int = 0
    advantage_collapse_rate: float = 0.0
    virtual_samples_injected: int = 0


class AVSPOAdvantageEstimator:
    """Estimates relative advantages for GRPO while preventing Advantage Collapse."""

    def __init__(
        self,
        variance_epsilon: float = 1e-6,
        virtual_bias_strength: float = 0.15,
        window_size: int = 50,
    ):
        """Args:

        variance_epsilon: Minimum std threshold below which rewards are
        considered collapsed. virtual_bias_strength: Magnitude of variance
        restoration when all samples are homogeneous. window_size: Rolling
        window for computing Advantage Collapse Rate (ACR).
        """
        self.variance_epsilon = variance_epsilon
        self.virtual_bias_strength = virtual_bias_strength
        self.window_size = window_size

        self.history_collapsed: List[bool] = []
        self.total_virtual_samples: int = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        process_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Compute normalized group-relative advantages for a group of G rollouts.

        Args:
            rewards: Tensor of shape (G,) representing binary outcome rewards.
            process_scores: Optional Tensor of shape (G,) representing PRM
              scores.

        Returns:
            Tuple of (advantages_tensor of shape (G,), was_collapsed_bool)
        """
        G = rewards.shape[0]
        if G <= 1:
            return torch.zeros_like(rewards), False

        rewards_float = rewards.to(torch.float32)
        std = torch.std(rewards_float, unbiased=False)
        mean = torch.mean(rewards_float)

        is_collapsed = bool(std.item() < self.variance_epsilon)
        self.history_collapsed.append(is_collapsed)
        if len(self.history_collapsed) > self.window_size:
            self.history_collapsed.pop(0)

        # Normal case: Non-zero variance exists across the group
        if not is_collapsed:
            adv = (rewards_float - mean) / (std + 1e-8)
            return adv, False

        # Advantage Collapse Case (All rewards are identical, e.g. all 0.0 or all 1.0)
        self.total_virtual_samples += 1

        if process_scores is not None and torch.std(process_scores) > self.variance_epsilon:
            # Method 1: Use Process Reward Model (PRM) variance to break the tie
            proc_std = torch.std(process_scores, unbiased=False)
            proc_mean = torch.mean(process_scores)
            adv = (process_scores - proc_mean) / (proc_std + 1e-8)
            return adv, True

        # Method 2: Adaptive Virtual Sample Injection
        # If all failed (mean == 0), inject virtual positive bias for shorter/cleaner sequences
        # If all passed (mean == 1), inject small relative differences
        # Construct synthetic anchor variance:
        # e.g., virtual bias gradient based on rank or small synthetic perturbation
        virtual_variance = torch.linspace(
            -self.virtual_bias_strength,
            self.virtual_bias_strength,
            steps=G,
            device=rewards.device,
        )
        augmented_rewards = rewards_float + virtual_variance
        aug_mean = torch.mean(augmented_rewards)
        aug_std = torch.std(augmented_rewards, unbiased=False)

        adv = (augmented_rewards - aug_mean) / (aug_std + 1e-8)
        return adv, True

    def get_stats(self) -> AVSPOStats:
        """Return current Advantage Collapse Rate and injection statistics."""
        total = len(self.history_collapsed)
        collapsed = sum(self.history_collapsed)
        acr = (collapsed / total) if total > 0 else 0.0

        return AVSPOStats(
            total_groups=total,
            collapsed_groups=collapsed,
            advantage_collapse_rate=acr,
            virtual_samples_injected=self.total_virtual_samples,
        )
