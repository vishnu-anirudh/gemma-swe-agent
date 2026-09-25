"""Unit tests for Dynamic Rank LoRA (DR-LoRA)."""

import torch
import torch.nn as nn
from src.peft.dr_lora import DRLoRALinear, DRLoRAManager
from src.peft.saliency import ModuleSaliencyTracker


def test_dr_lora_linear_forward():
    in_features = 32
    out_features = 16
    base_linear = nn.Linear(in_features, out_features)
    dr_layer = DRLoRALinear(
        base_layer=base_linear,
        max_rank=16,
        initial_rank=4,
        lora_alpha=8.0,
    )

    x = torch.randn(2, in_features)
    out = dr_layer(x)

    assert out.shape == (2, out_features)
    assert dr_layer.active_rank == 4
    assert dr_layer.base_layer.weight.requires_grad is False
    assert dr_layer.lora_A.requires_grad is True
    assert dr_layer.lora_B.requires_grad is True


def test_dr_lora_rank_adjustment():
    base_linear = nn.Linear(20, 10)
    dr_layer = DRLoRALinear(base_linear, max_rank=8, initial_rank=2)

    dr_layer.set_active_rank(6)
    assert dr_layer.active_rank == 6

    # Test clamping
    dr_layer.set_active_rank(20)
    assert dr_layer.active_rank == 8

    dr_layer.set_active_rank(0)
    assert dr_layer.active_rank == 1


def test_dr_lora_manager():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.v_proj = nn.Linear(16, 16)
            self.out_proj = nn.Linear(16, 8)

        def forward(self, x):
            return self.out_proj(self.q_proj(x) + self.v_proj(x))

    model = ToyModel()
    manager = DRLoRAManager(
        model=model,
        target_modules=["q_proj", "v_proj"],
        max_rank=16,
        initial_rank=4,
    )

    ranks = manager.get_rank_distribution()
    assert "q_proj" in ranks
    assert "v_proj" in ranks
    assert "out_proj" not in ranks
    assert ranks["q_proj"] == 4
    assert ranks["v_proj"] == 4

    # Simulate gradient update & saliency scoring
    x = torch.randn(2, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()

    manager.saliency_tracker.update_from_model(model)
    scores = manager.saliency_tracker.compute_saliency_scores()
    assert len(scores) > 0

    new_ranks = manager.reallocate_ranks(total_rank_budget=16)
    assert "q_proj" in new_ranks
    assert "v_proj" in new_ranks
