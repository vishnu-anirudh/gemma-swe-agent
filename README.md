# Autonomous SWE Gemma: Post-Training & Test-Time Scaling on Apple Silicon

[![CI / Unit Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](tests/)
[![Platform](https://img.shields.io/badge/hardware-Apple%20Silicon%20(M2%20Max%2064GB)-black.svg)](#hardware-acceleration)
[![Model](https://img.shields.io/badge/model-Google%20Gemma--2--2B--it-blue.svg)](https://huggingface.co/google/gemma-2-2b-it)
[![Evaluation](https://img.shields.io/badge/benchmark-SWE--bench%20Lite%20%26%20Verified-purple.svg)](https://www.swebench.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An end-to-end framework for turning **Google Gemma 2** into autonomous, repository-scale software engineering agents. Engineered and optimized from first principles for **Apple Silicon (MPS with Unified Memory)** using PyTorch and `uv`.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph SFT["Phase 1: Supervised Fine-Tuning (SFT)"]
        D[Curriculum Trajectories<br>Easy → Medium → Hard] --> M[Step-Level Error Masker<br>Masks exploration blunders to -100]
        M --> DRL[Dynamic Rank LoRA<br>Saliency-driven rank reallocation]
        DRL --> CKPT[(Fine-Tuned Adapter Checkpoint<br>adapter_model.pt)]
    end

    subgraph RLVR["Phase 2: Reinforcement Learning (RLVR)"]
        CKPT --> POL[Gemma Policy Model]
        POL --> ROLL[K Rollout Trajectories]
        ROLL --> DUAL[Dual-Suite Verifier<br>FAIL_TO_PASS + PASS_TO_PASS]
        ROLL --> PRM[Rubric Process Reward Model<br>Trajectory discipline & SRI format]
        DUAL & PRM --> AVSPO[AVSPO Advantage Normalization<br>Neutralizes zero-variance collapse]
        AVSPO --> GRPO[Critic-Free GRPO Loss<br>Memory-efficient on Unified RAM]
        GRPO -.->|Policy Updates| POL
    end

    subgraph TTS["Phase 3: Test-Time Scaling (PDR & Agent)"]
        POL --> AGENT[Multi-Turn Agent Loop<br>RepoTools: search, view, patch]
        POL --> PDR[Parallel-Distill-Refine (PDR)<br>K Parallel Candidates + Tournament Voting]
        AGENT & PDR --> WT[Docker & Git Worktree Sandboxes<br>Containerized or zero-copy ephemeral checkouts]
        WT --> SWE[SWE-bench Grader<br>Pass@1 & Pass@K Resolution]
    end
```

---

## 🔬 Core Innovations & Implementations

### 1. Dynamic Rank LoRA (`DR-LoRA`)
Standard PEFT applies uniform ranks across all projection layers, ignoring the specialized role of attention heads. DR-LoRA tracks runtime layer saliency:

$$S_{\ell, i} = \alpha \cdot f_{\text{freq}}(\ell, i) + \beta \cdot \|\nabla W_{\ell, i}\|_2$$

Applies **Total Variation (TV) smoothing** across adjacent transformer blocks and dynamically allocates rank capacity $r_{\ell} \in [r_{\min}, r_{\max}]$ to high-saliency projection layers while pruning dormant subspaces. Checkpoints preserve layer-specific active rank masks.
- **Source**: [`src/peft/dr_lora.py`](src/peft/dr_lora.py) & [`src/peft/saliency.py`](src/peft/saliency.py)

### 2. Step-Level Error Masking & Expanded Curriculum
Teacher models frequently produce intermediate syntax mistakes, failed validation commands, or invalid diff attempts before reaching the correct repair. Rather than penalizing or teaching the model to repeat errors, `StepLevelErrorMasker` computes surrogate loss only on successful transitions:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^T m_t \cdot \log P_\theta(y_t \mid y_{<t}, x), \quad m_t = \begin{cases} 1 & \text{state transition valid} \\ 0 & \text{masked syntax blunder} \end{cases}$$

Supports both 30-instance curated multi-tier curriculum generation (`--expanded`) and external JSONL trajectory datasets (`--dataset`).
- **Source**: [`src/data/masking.py`](src/data/masking.py) & [`src/data/curriculum.py`](src/data/curriculum.py)

### 3. Critic-Free GRPO with AVSPO Advantage Normalization
Eliminates value critic networks to fit training entirely within 64GB Unified Memory. Scaled to $G=8$ rollouts at $T=0.85$ for diverse candidate exploration.

Standard GRPO normalizes rewards across $G$ group rollouts:
$$A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

When binary verification rewards are homogeneous across a rollout group (all pass or all fail), $\sigma_R \to 0$, causing advantage collapse and vanishing policy gradients. **AVSPO (Adaptive Virtual Sample Policy Optimization)** detects zero-variance collapse via the Advantage Collapse Rate (ACR) and injects virtual prior bounds to preserve gradient signals.
- **Source**: [`src/rlvr/grpo.py`](src/rlvr/grpo.py) & [`src/rlvr/avspo.py`](src/rlvr/avspo.py)

### 4. Dual-Suite Sandbox Verifier & Official Docker Containers
Prevents model patches from passing regression suites at the expense of breaking existing functionality:
- **`FAIL_TO_PASS`**: Tests that must flip from failure to pass (bug fix verification).
- **`PASS_TO_PASS`**: Existing tests that must remain passing (anti-regression guarantee; reward dropped to $0.0$ if violated).
- **Docker Sandbox Integration**: Supports official SWE-bench evaluation containers (`swebench/sweb.eval.*`) with automatic fallback to local Git worktrees when Docker is inactive.
- **Source**: [`src/rlvr/verifier.py`](src/rlvr/verifier.py) & [`src/evaluation/repo_manager.py`](src/evaluation/repo_manager.py)

### 5. Search-and-Replace Infilling (SRI)
Standard unified diffs suffer from line-number hallucinations and context drift on small models. SRI parses exact search/replace target strings:
```text
<<<<<<< SEARCH
def old_function():
    return None
=======
def old_function():
    return True
>>>>>>> REPLACE
```
- **Source**: [`src/data/sri_formatter.py`](src/data/sri_formatter.py) & [`src/evaluation/patch_applicator.py`](src/evaluation/patch_applicator.py)

### 6. Parallel-Distill-Refine (PDR) Test-Time Scaling
Scales test-time compute by generating $K$ parallel rollout candidates, evaluating each via Rubric PRM and execution verification, distilling failure reasons, and conducting **Recursive Tournament Voting (RTV)** to select the optimal patch.
- **Source**: [`src/evaluation/parallel_distill_refine.py`](src/evaluation/parallel_distill_refine.py)

---

## 📂 Repository Layout

```
gemma_challenge/
├── configs/
│   ├── sft_dr_lora.yaml             # SFT curriculum & saliency rank configs
│   ├── rlvr_grpo.yaml               # GRPO (G=8, T=0.85) + AVSPO + PRM reinforcement learning
│   ├── swe_bench_eval.yaml          # Full SWE-bench evaluation config (limit=25)
│   └── swe_bench_eval_quick.yaml    # Quick sanity check eval config (limit=2)
├── src/
│   ├── agent/
│   │   ├── agent_loop.py            # MultiTurnAgent with stream telemetry
│   │   └── tools.py                 # search_code, view_file, list_dir, run_test
│   ├── data/
│   │   ├── sri_formatter.py         # SRI search/replace block parser & diff converter
│   │   ├── masking.py               # Step-level error masking for SFT loss
│   │   └── curriculum.py            # 30-instance expanded curriculum & JSONL loader
│   ├── peft/
│   │   ├── dr_lora.py               # Dynamic Rank LoRA layer & state manager
│   │   └── saliency.py              # Layer saliency tracking & TV smoothing
│   ├── rlvr/
│   │   ├── grpo.py                  # Critic-free GRPO policy trainer (G=8, T=0.85)
│   │   ├── avspo.py                 # AVSPO anti-collapse advantage normalization
│   │   ├── prm.py                   # Rubric Process Reward Model
│   │   └── verifier.py              # Dual-suite sandbox & DockerSandboxVerifier
│   ├── evaluation/
│   │   ├── swe_bench_loader.py      # SWE-bench dataset parser & filter
│   │   ├── patch_applicator.py      # Multi-format patch application engine
│   │   ├── repo_manager.py          # Git worktree sandbox manager
│   │   ├── parallel_distill_refine.py # PDR & Recursive Tournament Voting
│   │   ├── swe_bench_runner.py      # Automated benchmark grading harness
│   │   └── metrics.py               # Pass@1, Pass@K, and Rich table formatting
│   └── train.py                     # Unified CLI entry point
└── tests/                           # 25 Comprehensive unit tests
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
Install dependencies with [`uv`](https://github.com/astral-sh/uv):
```bash
# Clone the repository
git clone https://github.com/vishnu-anirudh/gemma-swe-agent.git
cd gemma-swe-agent

# Install dependencies and sync virtualenv
uv sync
```

### 2. Run the Verification Suite
All 25 unit tests execute in under 1 second:
```bash
uv run pytest -v
```

### 3. Supervised Fine-Tuning (Expanded Curriculum + DR-LoRA)
Fine-tune Gemma-2-2B-it with dynamic rank adaptation across the 30-instance expanded curriculum:
```bash
uv run python -m src.train --mode sft --config configs/sft_dr_lora.yaml --epochs 3 --expanded --checkpoint ./checkpoints/sft_dr_lora/adapter_model.pt
```

Or provide an external JSONL trajectory dataset:
```bash
uv run python -m src.train --mode sft --config configs/sft_dr_lora.yaml --epochs 3 --dataset path/to/trajectories.jsonl
```

### 4. Reinforcement Learning (Critic-Free GRPO with G=8 + AVSPO)
Optimize the policy using execution-grounded rewards with $G=8$ rollouts and save checkpoints:
```bash
uv run python -m src.train --mode rlvr --config configs/rlvr_grpo.yaml --checkpoint ./checkpoints/sft_dr_lora/adapter_model.pt --steps 3
```

### 5. Multi-Turn Interactive Agent Evaluation
Run autonomous multi-turn bug exploration and patching on SWE-bench Lite:
```bash
uv run python -m src.train --mode eval --config configs/swe_bench_eval_quick.yaml --agentic --checkpoint ./checkpoints/sft_dr_lora/adapter_model.pt
```

*(Optional: add `--docker` to run tests inside official SWE-bench Docker containers)*

### 6. Parallel-Distill-Refine (PDR) Test-Time Scaling
Run $K$-candidate parallel generation with Rubric PRM scoring and tournament selection:
```bash
uv run python -m src.train --mode eval --config configs/swe_bench_eval_quick.yaml --tts --k 4 --checkpoint ./checkpoints/sft_dr_lora/adapter_model.pt
```

---

## 💻 Hardware Acceleration (Apple Silicon)

This repository is optimized for Apple Silicon (M-Series processors) with PyTorch Metal Performance Shaders (`mps`):
- **Unified Memory Efficiency**: Eliminates duplicate CPU-to-GPU data copies.
- **bfloat16 Support**: Gemma-2 weights loaded natively in `torch.bfloat16`.
- **Zero-Critic Overhead**: Critic-free GRPO fits multi-rollout RLVR on a single Mac.

---

## 📜 License
Apache License 2.0. See [LICENSE](LICENSE) for details.
