"""Saliency metric computation for Dynamic Rank LoRA (DR-LoRA).

Measures layer/expert importance based on activation/routing frequency
and gradient magnitude (Frobenius norm).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ModuleSaliencyTracker:
    """Tracks gradient norm and activation statistics to compute module saliency."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """Args:

        alpha: Weight for activation/routing frequency.
        beta: Weight for gradient magnitude (Frobenius norm).
        """
        self.alpha = alpha
        self.beta = beta
        self.activation_counts: Dict[str, int] = {}
        self.gradient_norms: Dict[str, float] = {}
        self.total_tokens: int = 0

    def record_activation(self, module_name: str, count: int = 1) -> None:
        """Increment the activation/routing frequency counter for a module."""
        self.activation_counts[module_name] = (
            self.activation_counts.get(module_name, 0) + count
        )

    def record_gradient(self, module_name: str, tensor: torch.Tensor) -> None:
        """Compute and store the Frobenius norm of gradients."""
        if tensor.grad is not None:
            norm = torch.linalg.norm(tensor.grad.detach()).item()
            prev = self.gradient_norms.get(module_name, 0.0)
            self.gradient_norms[module_name] = max(prev, norm)

    def update_from_model(self, model: nn.Module) -> None:
        """Scan all named parameters in the model to accumulate gradient norms."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = torch.linalg.norm(param.grad.detach()).item()
                self.gradient_norms[name] = norm

    def compute_saliency_scores(self) -> Dict[str, float]:
        """Normalize statistics and calculate the final saliency score:

        S = alpha * norm_freq + beta * norm_grad
        """
        scores: Dict[str, float] = {}

        # Max frequency for normalization
        max_freq = max(self.activation_counts.values()) if self.activation_counts else 1.0
        max_grad = max(self.gradient_norms.values()) if self.gradient_norms else 1.0

        all_keys = set(self.activation_counts.keys()).union(self.gradient_norms.keys())

        for key in all_keys:
            norm_freq = self.activation_counts.get(key, 0) / max(max_freq, 1.0)
            norm_grad = self.gradient_norms.get(key, 0.0) / max(max_grad, 1e-8)
            scores[key] = (self.alpha * norm_freq) + (self.beta * norm_grad)

        return scores

    def reset_step(self) -> None:
        """Clear step-level gradient metrics."""
        self.gradient_norms.clear()
