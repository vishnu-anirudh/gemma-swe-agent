"""Dynamic Rank Low-Rank Adaptation (DR-LoRA).

Dynamically allocates and adjusts active LoRA rank capacity across model
layers and modules based on runtime saliency scoring.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.peft.saliency import ModuleSaliencyTracker


class DRLoRALinear(nn.Module):
    """Linear layer with dynamic active-rank Low-Rank Adaptation (LoRA)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        max_rank: int = 64,
        initial_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.max_rank = max_rank
        self.active_rank = min(initial_rank, max_rank)
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / self.active_rank

        # Freeze base linear layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA matrices allocated up to max_rank matching base layer device & dtype
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(max_rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, max_rank, device=device, dtype=dtype))

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform for A, zero for B to guarantee identity at step 0."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_active_rank(self, new_rank: int) -> None:
        """Update active rank dynamically."""
        clamped_rank = max(1, min(new_rank, self.max_rank))
        self.active_rank = clamped_rank
        self.scaling = self.lora_alpha / self.active_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward
        result = self.base_layer(x)

        # Dynamic low-rank forward using only active_rank slices
        # x: (..., in_features)
        # active_A: (active_rank, in_features)
        # active_B: (out_features, active_rank)
        active_A = self.lora_A[: self.active_rank, :]
        active_B = self.lora_B[:, : self.active_rank]

        dropped_x = self.dropout(x)
        lora_out = F.linear(dropped_x, active_A)  # (..., active_rank)
        lora_out = F.linear(lora_out, active_B)  # (..., out_features)

        return result + (lora_out * self.scaling)


class DRLoRAManager:
    """Manages injection of DRLoRALinear layers, saliency tracking, and rank reallocation."""

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        max_rank: int = 64,
        min_rank: int = 2,
        initial_rank: int = 8,
        lora_alpha: float = 16.0,
        smoothing_factor: float = 0.4,
    ):
        self.model = model
        self.target_modules = target_modules or ["q_proj", "v_proj", "gate_proj", "up_proj"]
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.initial_rank = initial_rank
        self.lora_alpha = lora_alpha
        self.smoothing_factor = smoothing_factor

        self.adapted_layers: Dict[str, DRLoRALinear] = {}
        self.saliency_tracker = ModuleSaliencyTracker()
        self._inject_adapters()

    def _inject_adapters(self) -> None:
        """Inject DRLoRALinear into matching submodules."""
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    if any(target in child_name for target in self.target_modules):
                        full_name = f"{name}.{child_name}" if name else child_name
                        adapted = DRLoRALinear(
                            base_layer=child,
                            max_rank=self.max_rank,
                            initial_rank=self.initial_rank,
                            lora_alpha=self.lora_alpha,
                        )
                        setattr(module, child_name, adapted)
                        self.adapted_layers[full_name] = adapted

    def reallocate_ranks(self, total_rank_budget: Optional[int] = None) -> Dict[str, int]:
        """Dynamically reallocate ranks to modules based on computed saliency scores.

        Applies Total Variation (TV) exponential smoothing to ensure stable rank
        transitions without wild oscillations across training steps.
        """
        scores = self.saliency_tracker.compute_saliency_scores()
        num_layers = len(self.adapted_layers)
        if num_layers == 0:
            return {}

        budget = total_rank_budget or (self.initial_rank * num_layers)

        # Map scores to adapted layer names
        layer_scores = {}
        for name in self.adapted_layers:
            matched_scores = [
                score for key, score in scores.items() if name in key
            ]
            layer_scores[name] = max(matched_scores) if matched_scores else 0.5

        # Normalize scores to allocate ranks proportionally
        total_score = sum(layer_scores.values()) or 1.0
        new_ranks: Dict[str, int] = {}

        for name, layer in self.adapted_layers.items():
            proportion = layer_scores[name] / total_score
            target_allocated = int(round(proportion * budget))

            # Apply TV / EMA smoothing with previous active rank:
            # r_new = (1 - gamma) * r_prev + gamma * r_target
            smoothed = round(
                (1.0 - self.smoothing_factor) * layer.active_rank
                + self.smoothing_factor * target_allocated
            )
            clamped = max(self.min_rank, min(smoothed, self.max_rank))
            layer.set_active_rank(clamped)
            new_ranks[name] = clamped

        return new_ranks

    def get_rank_distribution(self) -> Dict[str, int]:
        """Return the current active rank for all adapted layers."""
        return {name: layer.active_rank for name, layer in self.adapted_layers.items()}

    def save_adapters(self, save_path: str) -> None:
        """Save DR-LoRA adapter weights and rank distribution metadata."""
        import os
        from pathlib import Path

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "metadata": {
                "initial_rank": self.initial_rank,
                "max_rank": self.max_rank,
                "min_rank": self.min_rank,
                "lora_alpha": self.lora_alpha,
                "target_modules": self.target_modules,
            },
            "rank_distribution": self.get_rank_distribution(),
            "adapter_weights": {
                name: {
                    "lora_A": layer.lora_A.data.cpu(),
                    "lora_B": layer.lora_B.data.cpu(),
                }
                for name, layer in self.adapted_layers.items()
            },
        }
        torch.save(checkpoint_data, str(out_path))

    def load_adapters(self, load_path: str, device: Optional[torch.device] = None) -> None:
        """Load DR-LoRA adapter weights and restore active ranks."""
        data = torch.load(load_path, map_location="cpu")
        rank_dist = data.get("rank_distribution", {})
        weights = data.get("adapter_weights", {})

        for name, layer in self.adapted_layers.items():
            if name in rank_dist:
                layer.set_active_rank(rank_dist[name])
            if name in weights:
                target_dev = device or layer.lora_A.device
                target_dtype = layer.lora_A.dtype
                layer.lora_A.data.copy_(weights[name]["lora_A"].to(device=target_dev, dtype=target_dtype))
                layer.lora_B.data.copy_(weights[name]["lora_B"].to(device=target_dev, dtype=target_dtype))

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> DRLoRAManager:
        """Instantiate DRLoRAManager directly from a saved checkpoint file."""
        data = torch.load(checkpoint_path, map_location="cpu")
        meta = data.get("metadata", {})
        manager = cls(
            model=model,
            target_modules=meta.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            max_rank=meta.get("max_rank", 64),
            min_rank=meta.get("min_rank", 2),
            initial_rank=meta.get("initial_rank", 8),
            lora_alpha=meta.get("lora_alpha", 16.0),
        )
        manager.load_adapters(checkpoint_path, device=device)
        return manager


