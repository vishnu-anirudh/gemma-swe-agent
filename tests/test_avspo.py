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
