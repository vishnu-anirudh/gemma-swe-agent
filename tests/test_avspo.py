"""Unit tests for Adaptive Virtual Sample Policy Optimization (AVSPO)."""

import torch
from src.rlvr.avspo import AVSPOAdvantageEstimator


def test_avspo_normal_variance():
    estimator = AVSPOAdvantageEstimator()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    adv, collapsed = estimator.compute_advantages(rewards)
    assert not collapsed
    assert adv.shape == (4,)
    # Higher rewards should have positive advantages
    assert adv[0] > 0
    assert adv[1] < 0


def test_avspo_advantage_collapse_all_zeros():
    estimator = AVSPOAdvantageEstimator(virtual_bias_strength=0.2)
    # All rollouts failed (binary test suite failure)
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])

    adv, collapsed = estimator.compute_advantages(rewards)
    assert collapsed is True
    # Without AVSPO, adv would be all zeros (vanishing gradients).
    # With AVSPO, variance must be restored so gradients do not vanish:
    assert torch.std(adv) > 0.1
    assert adv.shape == (4,)


def test_avspo_advantage_collapse_all_ones():
    estimator = AVSPOAdvantageEstimator()
    # All rollouts passed
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])

    adv, collapsed = estimator.compute_advantages(rewards)
    assert collapsed is True
    assert torch.std(adv) > 0.1


def test_avspo_acr_monitoring():
    estimator = AVSPOAdvantageEstimator()
    # Process 3 collapsed batches and 1 normal batch
    estimator.compute_advantages(torch.tensor([0.0, 0.0, 0.0]))
    estimator.compute_advantages(torch.tensor([1.0, 1.0, 1.0]))
    estimator.compute_advantages(torch.tensor([0.0, 0.0, 0.0]))
    estimator.compute_advantages(torch.tensor([1.0, 0.0, 1.0]))

    stats = estimator.get_stats()
    assert stats.total_groups == 4
    assert stats.collapsed_groups == 3
    assert stats.advantage_collapse_rate == 0.75
    assert stats.virtual_samples_injected == 3


def test_avspo_structural_tie_breaker_penalizes_truncated_rollouts():
    estimator = AVSPOAdvantageEstimator(virtual_bias_strength=0.15)
    rollouts = [
        "No patch.",  # Short truncated output (9 chars)
        "<thought>\nFix bug.\n</thought>\n<<<<<<< SEARCH: a.py\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE",  # Complete concise SRI
        "<thought>\nFix bug.\n</thought>\n<<<<<<< SEARCH: a.py\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n" + ("# comment\n" * 20),  # Verbose SRI
        "Unformatted rambling text without any patch blocks at all " * 3,
    ]
    tie_scores = estimator.compute_structural_tie_breaker(rollouts)
    rewards = torch.zeros(4)
    adv, collapsed = estimator.compute_advantages(rewards, tie_breaker_scores=tie_scores)

    assert collapsed is True
    # Complete concise SRI (index 1) must receive the highest positive advantage
    assert torch.argmax(adv).item() == 1
    # Truncated output "No patch." (index 0) must receive the lowest negative advantage
    assert torch.argmin(adv).item() == 0
    assert adv[1] > 1.0
    assert adv[0] < -1.0

