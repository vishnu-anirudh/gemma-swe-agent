# Autonomous SWE Gemma: Post-Training on Apple Silicon

This repository implements the post-training methods and evaluation framework proposed in [`research.md`](./research.md) for turning Google's Gemma models into autonomous, repository-scale software engineering agents.

Engineered specifically for **Apple Silicon (M2 Max, 64GB Unified Memory)** using PyTorch with Metal Performance Shaders (`mps`).

---

## Key Algorithmic Implementations

### 1. Search-and-Replace Infilling (SRI)
- Eliminates brittle unified diffs and line-number hallucinations.
- Uses exact, unique string matching to inject edits into large files without breaking context.
- Implemented in [`src/data/sri_formatter.py`](./src/data/sri_formatter.py).

### 2. Step-Level Error Masking (SFT)
- Masks intermediate syntax errors, failed validation calls, and tool blunders from the surrogate loss (`label = -100`).
- Trains the model on the structural process of debugging without internalizing the teacher model's exploration mistakes.
- Implemented in [`src/data/masking.py`](./src/data/masking.py).

### 3. Dynamic Rank LoRA (DR-LoRA)
- Replaces uniform LoRA ranks ($r=64$ everywhere) with dynamic rank allocation.
- Tracks runtime module saliency scores:
  $$S_{\ell, i} = \alpha \cdot f_{\text{freq}}(\ell, i) + \beta \cdot \|\nabla W_{\ell, i}\|$$
- Dynamically expands rank capacity for critical coding and attention pathways while shrinking inactive modules.
- Implemented in [`src/peft/dr_lora.py`](./src/peft/dr_lora.py) and [`src/peft/saliency.py`](./src/peft/saliency.py).

### 4. Critic-Free GRPO with AVSPO (Anti-Advantage Collapse)
- Eliminates the need for a memory-intensive critic model, enabling RLVR on a single accelerator.
- **AVSPO (Adaptive Virtual Sample Policy Optimization)**: When binary test rewards are homogeneous across a group (e.g., all rollouts fail or all pass), $\sigma_R \to 0$ causing vanishing gradients. AVSPO monitors the Advantage Collapse Rate (ACR) and dynamically restores variance to keep the policy learning.
- Integrates a **Rubric Process Reward Model (PRM)** scoring fault localization, trajectory discipline, and SRI formatting validity.
- Implemented in [`src/rlvr/grpo.py`](./src/rlvr/grpo.py), [`src/rlvr/avspo.py`](./src/rlvr/avspo.py), and [`src/rlvr/prm.py`](./src/rlvr/prm.py).

### 5. SWE-bench Evaluation Suite
- Direct integration with official [`SWE-bench Lite`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) and [`SWE-bench Verified`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified).
- Isolated sandbox runner evaluating `FAIL_TO_PASS` and `PASS_TO_PASS` test targets.
- Computes standard `Pass@1` and `Pass@K` metrics.
- Implemented in [`src/evaluation/`](./src/evaluation/).

---

## Directory Structure

```
gemma_challenge/
├── research.md                  # Foundational research document
├── README.md                    # System documentation
├── pyproject.toml               # uv project configuration and dependencies
├── configs/
│   ├── sft_dr_lora.yaml         # SFT training hyperparameters & saliency settings
│   ├── rlvr_grpo.yaml           # GRPO + AVSPO + PRM reinforcement learning settings
│   └── swe_bench_eval.yaml      # SWE-bench Lite / Verified evaluation settings
├── src/
│   ├── data/
│   │   ├── sri_formatter.py     # SRI parser, applicator, and diff converter
│   │   └── masking.py           # Step-level error masking for SFT loss
│   ├── peft/
│   │   ├── dr_lora.py           # Dynamic Rank LoRA layer & budget allocator
│   │   └── saliency.py          # Saliency tracking (frequency + gradient norm)
│   ├── rlvr/
│   │   ├── grpo.py              # Critic-free Group Relative Policy Optimization trainer
│   │   ├── avspo.py             # AVSPO anti-collapse advantage normalization
│   │   ├── prm.py               # Rubric-based Process Reward Model
│   │   └── verifier.py          # Ephemeral subprocess execution verifier
│   ├── evaluation/
│   │   ├── swe_bench_loader.py  # SWE-bench dataset parser
│   │   ├── patch_applicator.py  # SRI and diff patch applier
│   │   ├── swe_bench_runner.py  # Test execution & resolution grader
│   │   └── metrics.py           # Pass@1, Pass@K, and rich summary tables
│   └── train.py                 # Unified CLI runner
└── tests/
    ├── test_sri.py              # Tests for SRI parsing and patching
    ├── test_dr_lora.py          # Tests for dynamic rank adaptation
    ├── test_avspo.py            # Tests for advantage collapse prevention
    └── test_swe_bench_pipeline.py # Tests for evaluation harness and metrics
```

---

## Quickstart & Usage with `uv`

### 1. Run Unit Tests
```bash
uv run pytest -v
```

### 2. Run Supervised Fine-Tuning (SFT with DR-LoRA)
```bash
uv run python -m src.train --mode sft --config configs/sft_dr_lora.yaml
```

### 3. Run Reinforcement Learning (RLVR with GRPO + AVSPO)
```bash
uv run python -m src.train --mode rlvr --config configs/rlvr_grpo.yaml
```

### 4. Evaluate on SWE-bench
```bash
uv run python -m src.train --mode eval --config configs/swe_bench_eval.yaml
```
