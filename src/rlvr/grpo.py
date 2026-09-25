"""Critic-Free Group Relative Policy Optimization (GRPO) Trainer with AVSPO.

Implements memory-efficient RLVR on Apple Silicon by sampling G rollouts per query,
scoring with verifiable test execution & PRM, and updating via clipped surrogate loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rlvr.avspo import AVSPOAdvantageEstimator


@dataclass
class GRPOTrainingConfig:
    """Hyperparameters for GRPO training."""

    group_size: int = 4  # G: number of rollouts per query
    epsilon_clip: float = 0.2
    kl_beta: float = 0.04
    learning_rate: float = 2e-5
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    prm_weight: float = 0.3  # Weight of Rubric PRM relative to binary execution reward


class GRPOTrainer:
    """Critic-free Group Relative Policy Optimization trainer."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[GRPOTrainingConfig] = None,
        reference_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or GRPOTrainingConfig()
        self.device = (
            device
            or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        )
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.avspo = AVSPOAdvantageEstimator()

        # Reference model for KL penalty (frozen)
        if reference_model is None:
            self.ref_model = copy.deepcopy(model).to(self.device)
        else:
            self.ref_model = reference_model.to(self.device)

        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.optimizer = optimizer or torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

    def generate_group_rollouts(
        self,
        prompt_text: str,
        group_size: Optional[int] = None,
    ) -> List[str]:
        """Sample G independent candidate responses from current policy."""
        G = group_size or self.config.group_size
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=G,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        rollouts = []
        for i in range(G):
            gen_tokens = output_tokens[i][prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            rollouts.append(text)

        return rollouts

    def compute_log_probs(
        self,
        model: nn.Module,
        prompt_text: str,
        response_text: str,
    ) -> torch.Tensor:
        """Compute the sum of log probabilities of response tokens given prompt."""
        full_text = prompt_text + response_text
        encoded_full = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        encoded_prompt = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        input_ids = encoded_full["input_ids"]
        prompt_len = encoded_prompt["input_ids"].shape[1]

        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # (1, seq_len - 1, vocab)
        target_ids = input_ids[:, 1:]  # (1, seq_len - 1)

        # Only evaluate response tokens
        response_logits = logits[:, prompt_len - 1 :, :]
        response_targets = target_ids[:, prompt_len - 1 :]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=response_targets.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()

    def train_step(
        self,
        prompt_text: str,
        rollouts: List[str],
        execution_rewards: List[float],
        process_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Perform one critic-free GRPO update step on the group of rollouts."""
        self.model.train()
        G = len(rollouts)
        raw_rewards = torch.tensor(execution_rewards, dtype=torch.float32, device=self.device)

        if process_scores is not None:
            proc_tensor = torch.tensor(process_scores, dtype=torch.float32, device=self.device)
            # Combined composite reward
            combined_rewards = (
                (1.0 - self.config.prm_weight) * raw_rewards
                + self.config.prm_weight * proc_tensor
            )
        else:
            proc_tensor = None
            combined_rewards = raw_rewards

        # Compute advantages using AVSPO (combats collapse if all passed/failed)
        advantages, was_collapsed = self.avspo.compute_advantages(
            combined_rewards, process_scores=proc_tensor
        )

        loss_list = []
        kl_list = []

        self.optimizer.zero_grad()

        for i in range(G):
            resp = rollouts[i]
            adv = advantages[i]

            # Current policy log prob
            cur_log_prob = self.compute_log_probs(self.model, prompt_text, resp)

            # Reference policy log prob for KL penalty
            with torch.no_grad():
                ref_log_prob = self.compute_log_probs(self.ref_model, prompt_text, resp)

            ratio = torch.exp(cur_log_prob - cur_log_prob.detach())  # approx ratio for step
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.config.epsilon_clip, 1.0 + self.config.epsilon_clip) * adv
            policy_loss = -torch.min(surr1, surr2)

            kl_div = cur_log_prob - ref_log_prob
            kl_penalty = self.config.kl_beta * kl_div

            total_sample_loss = policy_loss + kl_penalty
            loss_list.append(policy_loss.item())
            kl_list.append(kl_penalty.item())

            # Backprop per sample to conserve peak memory on Apple Silicon
            (total_sample_loss / G).backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        stats = self.avspo.get_stats()
        return {
            "mean_loss": sum(loss_list) / G,
            "mean_reward": float(raw_rewards.mean().item()),
            "mean_kl": sum(kl_list) / G,
            "was_collapsed": float(was_collapsed),
            "advantage_collapse_rate": stats.advantage_collapse_rate,
        }
