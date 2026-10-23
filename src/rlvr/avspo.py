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
        tie_breaker_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Compute normalized group-relative advantages for a group of G rollouts.

        Implements a two-tier collapse-recovery estimator when binary execution
        rewards exhibit zero variance (std(R) < variance_epsilon):
          - Tier 1 (Rubric PRM Substitution): If dense process_scores P_i have
            non-zero variance, substitute P_i into both numerator and denominator:
            A_i = (P_i - mean(P)) / (std(P) + eps).
          - Tier 2 (Virtual Reward Perturbation): Perturb the reward vector itself
            via R_tilde_i = R_i + Delta_i (where sum_i Delta_i = 0), aligned with
            tie_breaker_scores (e.g., negative sequence length) when provided, so
            both numerator (R_tilde_i - mean(R_tilde)) and denominator std(R_tilde)
            are strictly non-zero.

        Args:
            rewards: Tensor of shape (G,) representing outcome rewards.
            process_scores: Optional Tensor of shape (G,) representing PRM scores.
            tie_breaker_scores: Optional Tensor of shape (G,) used to rank virtual
              perturbations (higher score receives positive perturbation).

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

        if process_scores is not None and torch.std(process_scores, unbiased=False) > self.variance_epsilon:
            # Tier 1: Use Process Reward Model (PRM) variance in numerator and denominator
            proc_float = process_scores.to(torch.float32)
            proc_std = torch.std(proc_float, unbiased=False)
            proc_mean = torch.mean(proc_float)
            adv = (proc_float - proc_mean) / (proc_std + 1e-8)
            return adv, True

        # Tier 2: Adaptive Virtual Sample Reward Perturbation (R_tilde_i = R_i + Delta_i)
        base_perturbation = torch.linspace(
            -self.virtual_bias_strength,
            self.virtual_bias_strength,
            steps=G,
            device=rewards.device,
        )
        if tie_breaker_scores is not None and tie_breaker_scores.shape[0] == G:
            # Assign higher virtual perturbation to higher tie-breaker scores (e.g., shorter length)
            sorted_indices = torch.argsort(tie_breaker_scores.to(rewards.device))
            virtual_perturbation = torch.zeros_like(base_perturbation)
            virtual_perturbation[sorted_indices] = base_perturbation
        else:
            virtual_perturbation = base_perturbation

        augmented_rewards = rewards_float + virtual_perturbation
        aug_mean = torch.mean(augmented_rewards)
        aug_std = torch.std(augmented_rewards, unbiased=False)

        adv = (augmented_rewards - aug_mean) / (aug_std + 1e-8)
        return adv, True

    @staticmethod
    def compute_structural_tie_breaker(
        rollouts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute structural-completeness-gated tie-breaker scores s_i for Tier-2 AVSPO.

        Prevents naive length tie-breaking from rewarding truncated or incomplete outputs:
          s_i = I[valid_sri(y_i)] + 0.25 * I[closed_thought(y_i)]
                - 1.0 * I[len(y_i) < 32] - 1e-4 * len(y_i)
        """
        scores: List[float] = []
        for text in rollouts:
            stripped = text.strip()
            length = len(stripped)
            has_sri = (
                "<<<<<<< SEARCH" in stripped
                and "=======" in stripped
                and ">>>>>>> REPLACE" in stripped
            )
            has_thought = "<thought>" in stripped and "</thought>" in stripped
            is_truncated = length < 32

            score = (
                (1.0 if has_sri else 0.0)
                + (0.25 if has_thought else 0.0)
                - (1.0 if is_truncated else 0.0)
                - 1e-4 * float(length)
            )
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32, device=device)

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
