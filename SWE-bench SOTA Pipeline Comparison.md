# Post-Training and Agentic Scaffolding for Sub-3B Software Engineering Models on Apple Silicon: A Small-Sample Local-Hardware Feasibility and Component Ablation Study ($N=4$ Sandbox, $N=25$ Zero-Shot Slice)

---

## 1. Abstract and Research Questions

### 1.1 Abstract
Autonomous resolution of repository-level software engineering issues on SWE-bench has predominantly relied on frontier proprietary models (>100B parameters) or mid-sized open-weight models (32B–72B) deployed on multi-GPU server clusters. Adapting and evaluating compact language models ($\le 3\text{B}$ parameters) locally on consumer Apple Silicon hardware introduces four practical bottlenecks: (1) high syntactic patch-application failure rates when generating standard unified `git diff` hunks with fragile line-number headers, (2) heterogeneous rank sensitivity across attention and multi-layer perceptron (MLP) projections under tight parameter budgets, (3) imitation of intermediate tool-invocation blunders during trajectory supervised fine-tuning (SFT), and (4) **Advantage Collapse** during Group Relative Policy Optimization (GRPO) when binary unit-test verifiers return identical rewards ($\mathbf{R} = \mathbf{0}$ or $\mathbf{R} = \mathbf{1}$) across all rollouts in a group.

In this **small-sample local-hardware feasibility and component-ablation study**, we implement and audit a modular post-training and inference pipeline for `google/gemma-2-2b-it` (2.61B parameters) running natively on a single Apple M2 Max workstation (64 GB unified memory, PyTorch `mps`). Our implementation combines: (i) a **Search-and-Replace Identifier (SRI)** edit interface paired with a 6-tier fuzzy AST/window patch applicator; (ii) a local discrete heuristic adaptation of **Dynamic Rank LoRA (DR-LoRA)** with invariant rank scaling and exponential moving average (EMA) step-transition smoothing across 182 linear submodules; (iii) **Step-Level Error Masking** (`ignore_index = -100`) to exclude intermediate tool errors and environment observations from the SFT loss; (iv) **Critic-Free GRPO** paired with a **Two-Tier Adaptive Virtual Sample Policy Optimization (AVSPO)** advantage estimator and a **zero-leakage observable-context Rubric Process Reward Model (PRM)**; and (v) a multi-turn repository localization agent (`view_file`, `grep_search`, `list_files`) with **2-round test-guided failure distillation and Parallel-Distill-Refine (PDR)** test-time candidate selection.

All experimental measurements, seed-level logs (`seeds = [42, 43, 44]`), exact 95% Clopper-Pearson binomial confidence intervals (CI), and checkpoint tensor audits are archived in the repository at [`results/ablation_and_audit_log.json`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/results/ablation_and_audit_log.json) and reproducible via [`scripts/run_ablations_and_audit.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/scripts/run_ablations_and_audit.py):
1. **Zero-Shot Syntactic Baseline (`N = 25` SWE-bench Lite slice):** Prompting `gemma-2-2b-it` with the SRI grammar achieves **84.0% (`21/25`, 95% CI: `[63.9%, 95.5%]`) syntactic SRI validity** at **3.80 s/instance** single-turn generation latency, but **0.0% (`0/25`, 95% CI: `[0.0%, 13.7%]`) issue resolution** without repository file access.
2. **DR-LoRA SFT & Step-Level Error Masking (`N = 40` curriculum trajectories; 3-seed controlled ablations):** Across 3 epochs on `gemma-2-2b-it` (`10,627,072` active adapter parameters; `0.406%` of base weights), masked SFT loss decreases from **3.0056 to 1.0124**, concentrating higher active rank in `v_proj` (`8.88`) and MLP `up_proj` (`8.88`) / `gate_proj` (`8.62`) relative to `q_proj` (`6.50`) and `k_proj` (`6.46`). In 3-seed controlled `ToyLM` ablations, DR-LoRA ($\gamma=0.4$) reduces final cross-entropy loss by **7.48%** relative to Uniform LoRA ($r=8$), while Step-Level Error Masking suppresses the token probability of erroneous tool calls by **45.7x** (`18.21% -> 0.40%`, NLL `1.7037 -> 5.5455`).
3. **Multi-Group AVSPO Estimator & Executable Unit-Test Validation (`5` estimators $\times$ `M = 10` bug task groups $\times$ `3` seeds = `30` group updates/estimator):** On homogeneous zero-reward groups ($\mathbf{R} = [0,0,0,0]^\top$), Vanilla GRPO suffers complete gradient collapse ($\|\nabla_\theta \mathcal{L}\|_2 = 0.00$), resolving **`8/30` (`26.67%` $\pm$ `4.71%`)** downstream unit tests via base initialization. Crucially, **naive length tie-breaking ($s_i = -\text{len}(y_i)$) is pathological**: it rewards truncated failure strings (`"No patch."`, 9 chars), driving post-update truncated output probability to **`96.67%`** and reducing executable unit-test resolution to **`0/30` (`0.00%`)**. **Structural-Completeness-Gated Tie-Breaking** (Tier 2) reduces truncated selection to `3.33%` (`86.67%` valid SRI selection), while **Zero-Leakage Observable-Context Rubric PRM Substitution** (Tier 1) eliminates truncated outputs (`0.00%`), achieves **`96.67%` (`29/30`) valid SRI selection**, and increases downstream executable unit-test resolution to **`20/30` (`66.67%` $\pm$ `4.71%`; +40.0 percentage points over Vanilla GRPO)**.
4. **Paired End-to-End Sandbox Verification & Component-Removal Audit (`N = 4` held-out SWE-bench Lite tasks across 7 configurations `C0`–`C5`):** On the paired 4-task testbed (`django__django-10914`, `django__django-10924`, `astropy__astropy-12907`, `astropy__astropy-14182`), 1-round single-pass synthesis with Multi-Turn Localization + Step-Level Error Masking + DR-LoRA (`C4a`) improves clean patch application from **`0/4` (`0.0%`)** to **`3/4` (`75.0%`, 95% CI: `[19.4%, 99.4%]`)** and resolves **`1/4` (`25.0%`, `django-10914`)**. Adding 3 steps of local GRPO-AVSPO (`C4b`) validates critic-free MPS memory (`12.60 GB`) and `0.00` Advantage Collapse Rate (`ACR`) and improves $T=0.7$ multi-sample SRI formatting stability (`81.3% -> 100.0%`), **but does not resolve additional tasks in 1-round single-pass synthesis (`1/4` vs. `1/4`)**. Adding **2-Round Test-Guided Failure Distillation and PDR ($K=4$, `C5`)** resolves `django-10924` on Round 2 by distilling Round 1's `TypeError: 'str' object is not callable` traceback, raising end-to-end resolution to **`2/4` (`50.0%`, 95% CI: `[6.8%, 93.2%]`; `2/2` on the Django subset)**. Given the small sample size ($N=4$), these findings are presented strictly as a **local-hardware feasibility and ablation study** rather than a claim of benchmark competitiveness across SWE-bench Lite ($N=300$).

---

### 1.2 Research Questions
* **RQ1 (Syntactic Validity vs. Repository Grounding):** *To what extent does enforcing a Search-and-Replace Identifier (SRI) grammar achieve syntactic patch parsability in a 2.61B-parameter model ($N=25$), and why does syntactic validity without multi-turn file inspection yield 0% issue resolution?*
* **RQ2 (Dynamic Low-Rank Capacity Allocation & Trajectory Error Masking):** *How does saliency-driven rank reallocation (DR-LoRA) distribute capacity across attention and MLP projections in `gemma-2-2b-it` compared to uniform LoRA, and how do DR-LoRA and Step-Level Error Masking affect intermediate optimization metrics and downstream 1-round patch application?*
* **RQ3 (Preventing Pathological Tie-Breaking & Zero-Leakage Localization in Collapse-Resistant RLVR):** *When binary unit-test rewards collapse to zero variance ($\sigma_{\mathbf{R}} = 0$), why does naive sequence-length tie-breaking fail across diverse bug groups ($M=10$), how can localization process rewards ($S_{\text{loc}}$) be computed strictly from observable context without gold-patch leakage, and how do Tier-2 and Tier-1 estimators affect executable unit-test resolution?*
* **RQ4 (Isolated Component Contributions: 1-Round Synthesis vs. 3-Step GRPO vs. 2-Round Test-Guided PDR):** *When evaluated as a paired study on the same $N=4$ repository sandbox tasks, what is the separate contribution of 1-round DR-LoRA SFT (`C4a`), 3-step GRPO-AVSPO (`C4b`), and 2-round test-guided failure distillation / PDR (`C5`)?*

---

## 2. Related Work, Bibliographic Verification, and Algorithmic Fidelity Audit

### 2.1 Repository-Level Software Engineering Benchmarks
**SWE-bench** (Jimenez et al., 2024) established execution-based evaluation on 2,294 real GitHub issue-commit pairs across 12 Python repositories, alongside the 300-instance **SWE-bench Lite** subset. Subsequent audits introduced **SWE-bench Verified** (OpenAI, 2024), a 500-instance human-filtered split removing underspecified unit tests, and **SWE-bench+** (Aleithan et al., 2024), which documented solution leakage in issue descriptions and brittle test patches.

### 2.2 Agentic Scaffolding and Search-and-Replace Edit Interfaces
**SWE-agent** (Yang et al., 2024) demonstrated that replacing raw bash/diff generation with a structured Agent-Computer Interface (ACI) substantially improves patch validity. **Agentless** (Xia et al., 2024) showed that hierarchical localization followed by `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` block generation and regression filtering achieves strong performance without complex autonomous loops. **OpenHands** (Wang et al., 2024) generalized multi-turn sandboxed execution for software agents, while **SWE-Lego** (Tao et al., 2026) demonstrated modular trajectory SFT with step-level error masking and difficulty curricula on 8B–32B models.

### 2.3 Parameter-Efficient Fine-Tuning and Verifiable Reinforcement Learning
Standard **LoRA** (Hu et al., 2022) assigns a fixed rank $r$ across all adapted matrices. **AdaLoRA** (Zhang et al., 2023) introduced sensitivity-based dynamic rank allocation over SVD-parameterized adapters, and **ARD-LoRA** (Shinwari & Usama, 2025) formulated dynamic rank allocation using continuous scaling factors optimized under differentiable $\ell_1$ sparsity and Total Variation (TV) regularization. In reinforcement post-training, **GRPO** (Shao et al., 2024; DeepSeek-AI, 2025) eliminated the PPO critic network (Schulman et al., 2017) via within-group reward normalization. To address sparse binary outcome rewards, **Process Reward Models (PRMs)** (Lightman et al., 2023) provide step-level supervision, **AVSPO** (He et al., 2025/2026) diagnosed Advantage Collapse in GRPO via the Advantage Collapse Rate (ACR) and introduced adaptive virtual sample stabilization, and **Test-Time Compute Scaling** (Snell et al., 2024) leverages multi-candidate sampling and verifier selection at inference time.

### 2.4 Contextual Comparison and Algorithmic Fidelity Audit (Addressing Reviewer Issue 6)
Table 1 contrasts published cloud/cluster pipelines with our local Apple Silicon feasibility study. Because our local implementations simplify or adapt several published algorithms to operate within a 64 GB unified-memory budget without auxiliary neural critics, Table 2 provides an explicit **Algorithmic Fidelity Audit** comparing each foundational paper's exact method against our local repository implementation.

| System / Architecture | Base Model & Scale | Hardware Target | Adaptation & Post-Training | Edit Interface | Reported Evaluation Scope & Resolution | Source Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-agent** (Yang et al., 2024) | GPT-4 Turbo / Claude 3.5 Sonnet (>100B) | Cloud API | Prompting + ACI Scaffolding | Windowed file editor | 12.5%–33.6% on SWE-bench Full / Lite ($N=300$) | Published Literature |
| **Agentless** (Xia et al., 2024) | GPT-4o (>100B) | Cloud API | Hierarchical Localization + Sampling | Search/Replace blocks | 27.3%–32.0% on SWE-bench Lite ($N=300$) | Published Literature |
| **SWE-Lego** (Tao et al., 2026) | Qwen2.5-Coder / Qwen3 (8B–32B) | $8\times$ H100 (80 GB) | Full / LoRA SFT with Error Masking | Search/Replace blocks | 42.0%–52.6% on SWE-bench Verified ($N=500$) | Published Literature |
| **SWE-Gemma (`C0-Lite`: Zero-Shot Base)** | `google/gemma-2-2b-it` (2.61B) | 1x Apple M2 Max (64 GB) | Frozen Pre-trained Base (`bfloat16`) | SRI (`SEARCH`/`REPLACE`) | **84.0% (`21/25`)** valid SRI; **0.0% (`0/25`)** resolved (Lite slice) | **Measured (This Study)** |
| **SWE-Gemma (`C4a`/`C4b`: 1-Round SFT / +3-Step GRPO)** | `google/gemma-2-2b-it` (2.61B + 10.6M active LoRA) | 1x Apple M2 Max (64 GB) | DR-LoRA SFT + Error Masking ($\pm$ 3-step GRPO-AVSPO), 1-round synthesis | SRI + 6-Tier Fuzzy Matcher | **100.0% (`4/4`)** valid SRI; **75.0% (`3/4`)** patch applied; **25.0% (`1/4`)** resolved (95% CI: `[0.6%, 80.6%]`) | **Measured (This Study)** |
| **SWE-Gemma (`C5`: Full Pipeline + 2-Round PDR)** | `google/gemma-2-2b-it` (2.61B + 10.6M active LoRA) | 1x Apple M2 Max (64 GB) | `C4b` + 2-Round Test-Guided Failure Distillation & PDR ($K=4$) | SRI + 6-Tier Fuzzy Matcher | **100.0% (`4/4`)** valid SRI; **75.0% (`3/4`)** patch applied; **50.0% (`2/4`)** resolved (95% CI: `[6.8%, 93.2%]`) | **Measured (This Study)** |

#### Table 2: Algorithmic Fidelity Audit — Published Formulations vs. Local Repository Implementations
| Subsystem & Code Path | Verified Published Reference | Exact Formulation in Published Paper | Local Adaptation Implemented in This Repository | Fidelity & Scope Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Search/Replace Edit Interface (SRI)** ([`src/data/sri_formatter.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/sri_formatter.py)) | **Agentless** (Xia et al., 2024, `arXiv:2407.01489`); **SWE-agent** (Yang et al., 2024, `arXiv:2405.15793`) | `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` code replacement blocks to bypass line-number calculation errors in unified diffs. | Adds an explicit file-path header (`<<<<<<< SEARCH: path/to/file.py`) and a **6-tier deterministic fallback parser** (line-number stripping, whitespace normalization, indentation delta $\Delta_{\text{indent}}$ preservation, AST anchor scoping, `difflib` window matching). | **Faithful Extension** for sub-3B formatting noise. |
| **Dynamic Rank LoRA (`DRLoRAManager`)** ([`src/peft/dr_lora.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/peft/dr_lora.py), [`src/peft/saliency.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/peft/saliency.py)) | **AdaLoRA** (Zhang et al., 2023, `arXiv:2303.10512`); **ARD-LoRA** (Shinwari & Usama, 2025, `arXiv:2506.18267`) | ARD-LoRA learns continuous per-head scaling factors via a differentiable meta-objective combining $\ell_1$ sparsity and a Total Variation (TV) penalty; AdaLoRA prunes SVD triplets via sensitivity scoring. | Implements a **discrete heuristic rank allocator** over pre-allocated `max_rank=64` slices (`A[:r_m, :]`, `B[:, :r_m]`) combining Taylor gradient sensitivity + singular-value energy with **step-transition EMA smoothing** ($\gamma=0.4$) and invariant forward scaling $\eta = \alpha / r_0$. | **Simplified Discrete Heuristic** inspired by AdaLoRA & ARD-LoRA (avoids bi-level meta-gradients on MPS). |
| **Step-Level Error Masking & Curriculum SFT** ([`src/data/masking.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/masking.py), [`src/data/curriculum.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/curriculum.py)) | **SWE-Lego** (Tao et al., 2026, `arXiv:2601.01426`) | Masks intermediate erroneous agent turns (`labels = -100`) and orders multi-turn trajectories from easy to hard during SFT on 8B–32B models. | Directly implements `StepLevelErrorMasker` setting `labels = -100` on system prompts, tool observations, and assistant turns flagged with `is_error=True`, paired with 3-stage curriculum sorting (`easy -> medium -> hard`) on $N=40$ local trajectories. | **Direct Local Implementation** at compact scale ($N=40$ trajectories, 2.61B model). |
| **Critic-Free GRPO + Two-Tier AVSPO + Zero-Leakage PRM** ([`src/rlvr/grpo.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/grpo.py), [`src/rlvr/avspo.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/avspo.py), [`src/rlvr/prm.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/prm.py)) | **GRPO** (Shao et al., 2024, `arXiv:2402.03300`); **AVSPO** (He et al., 2025/2026, `arXiv:2605.21125`); **PRM** (Lightman et al., 2023, `arXiv:2305.20050`) | AVSPO defines Advantage Collapse Rate (ACR) when group reward variance vanishes ($\sigma_{\mathbf{R}} \to 0$) and injects virtual sample variance / instance weighting to prevent policy stagnation. | Adopts AVSPO's ACR diagnostic and virtual variance concept, adapting it into a **two-tier SWE advantage estimator**: Tier 1 substitutes a rule-based, zero-leakage observable-context `RubricProcessRewardModel` ($P_i$), and Tier 2 applies **structural-completeness-gated** linear reward perturbations ($\tilde{R}_i = R_i + \Delta_{\pi^{-1}(i)}$). | **Domain-Specific Adaptation** for zero-extra-memory SWE RLVR on Apple Silicon. |
| **Test-Guided Failure Distillation & PDR ($K=4$)** ([`src/agent/agent_loop.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/agent/agent_loop.py), [`src/evaluation/parallel_distill_refine.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/evaluation/parallel_distill_refine.py)) | **Test-Time Compute Scaling** (Snell et al., 2024, `arXiv:2408.03314`) | Scales inference compute via parallel candidate generation, verifier scoring, and iterative refinement conditioned on execution feedback. | Implements parallel temperature diversity ($T \in \{0.2, 0.4, 0.6, 0.8\}$), git worktree dual-suite verification (`FAIL_TO_PASS` + `PASS_TO_PASS`), traceback distillation (`distill_failure_summary`), and Round-2 repair synthesis. | **Direct Instantiation** for local git-worktree test harnesses. |

---

## 3. Methodology and Mathematical Formulation

### 3.1 Search-and-Replace Identifier (SRI) Grammar and 6-Tier Fuzzy Patch Applicator
To avoid unified diff line-number arithmetic errors, the model is prompted and trained to emit edits in the SRI grammar ([`src/data/sri_formatter.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/sri_formatter.py)):

```text
<thought>
1. Localize target file and inspect exact lines via view_file.
2. Formulate minimal replacement preserving surrounding invariants.
</thought>
<final_patch>
<<<<<<< SEARCH: relative/path/to/file.py
exact_existing_lines
=======
replacement_lines
>>>>>>> REPLACE
</final_patch>
```

When applying an extracted `SRIBlock` to a repository file $C$, `SRIFormatter.apply_to_content` ([`src/data/sri_formatter.py#L112-L280`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/sri_formatter.py#L112-L280)) executes a deterministic **6-tier matching cascade**:
1. **Tier 1 (Exact Unique Substring Match):** Requires $\text{count}(C, \text{search\_content}) = 1$.
2. **Tier 2 (Line-Number Prefix Stripping):** Removes accidentally copied viewer line numbers (`^\s*\d+\s*[:|]\s?`).
3. **Tier 3 (Trailing Whitespace Normalization):** Strips trailing spaces/tabs per line before matching.
4. **Tier 4 (Indentation-Agnostic Line Sequence Matching):** Matches stripped line sequences and re-indents `replace_content` by the target block's leading whitespace offset $\Delta_{\text{indent}}$.
5. **Tier 5 (AST Anchor-Scoped Disambiguation):** Resolves globally non-unique search lines by scoping to enclosing `class` or `def` headers present in the block.
6. **Tier 6 (Sliding-Window Fuzzy Sequence Matching):** Applies `difflib.SequenceMatcher` with threshold $\tau \ge 0.75$ while preserving the target block's base indentation offset $\Delta_{\text{indent}}$ across all replaced lines.

---

### 3.2 Dynamic Rank LoRA (DR-LoRA) with Invariant Forward Scaling
For each adapted linear projection $m \in \{1, \dots, M\}$ with frozen weight $W_0^{(m)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, `DRLoRALinear` ([`src/peft/dr_lora.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/peft/dr_lora.py)) pre-allocates $A^{(m)} \in \mathbb{R}^{r_{\max} \times d_{\text{in}}}$ and $B^{(m)} \in \mathbb{R}^{d_{\text{out}} \times r_{\max}}$ ($r_{\max} = 64$) and dynamically slices the active rank $r_m \in [r_{\min}, r_{\max}]$ ($r_{\min}=2, r_0=8$):
$$h^{(m)}(x) = W_0^{(m)} x + \frac{\alpha}{r_0} B^{(m)}_{:, \, 1:r_m} \left( A^{(m)}_{1:r_m, \, :} x \right)$$
Holding $\eta = \frac{\alpha}{r_0}$ invariant across rank updates prevents logit magnitude jumps when $r_m$ changes at reallocation boundaries. Module saliency $S_m^{(t)}$ combines first-order Taylor gradient sensitivity and singular-value energy ([`src/peft/saliency.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/peft/saliency.py)), and active ranks are updated under global budget $B_{\text{rank}} = M \cdot r_0$ with step-transition EMA smoothing factor $\gamma = 0.4$:
$$\tilde{r}_m^{(t)} = \text{round}\!\left( \frac{S_m^{(t)}}{\sum_{j=1}^M S_j^{(t)}} \cdot B_{\text{rank}} \right), \qquad r_m^{(t)} = \text{clamp}\!\left( \text{round}\!\left( (1 - \gamma) r_m^{(t-1)} + \gamma \tilde{r}_m^{(t)} \right), \, r_{\min}, \, r_{\max} \right)$$

---

### 3.3 Step-Level Error Masking for Multi-Turn Trajectory SFT
Given an exploration trajectory $\tau = (s_{\text{sys}}, (a_1, o_1, e_1), \dots, (a_K, o_K, e_K), a_{\text{patch}})$, `StepLevelErrorMasker` ([`src/data/masking.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/data/masking.py)) sets target labels to `ignore_index = -100` ($m_t = 0$) on system prompts $s_{\text{sys}}$, environment tool outputs $o_k$, and any intermediate action $a_k$ that triggered an execution or syntax error ($e_k = 1$):
$$\mathcal{L}_{\text{Masked-SFT}}(\theta) = -\frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \log \pi_\theta(x_t \mid x_{<t}), \qquad m_t = \mathbb{I}\!\left[ x_t \in a_{\text{patch}} \lor (x_t \in a_k \land e_k = 0) \right]$$

---

### 3.4 Critic-Free GRPO, Zero-Leakage Observable-Context PRM, and Two-Tier AVSPO

#### Why Denominator-Only Perturbation Cannot Resolve Advantage Collapse
In standard GRPO (Shao et al., 2024), group-relative advantages for $G$ rollouts $\{y_1, \dots, y_G\}$ with scalar rewards $\mathbf{R} = (R_1, \dots, R_G)^\top$ are computed as:
$$\hat{A}_i^{\text{GRPO}} = \frac{R_i - \mu_{\mathbf{R}}}{\sigma_{\mathbf{R}} + \epsilon}$$
When all $G$ rollouts fail ($\mathbf{R} = \mathbf{0}$) or all pass ($\mathbf{R} = \mathbf{1}$), $R_i - \mu_{\mathbf{R}} = 0$ for every $i \in \{1, \dots, G\}$. Perturbing only the denominator $\sigma_{\mathbf{R}}$ still yields $\frac{0}{\sigma_{\text{perturbed}}} = 0$. Therefore, a valid collapse-recovery estimator must modify **both the numerator and denominator**.

#### Why Naive Length Tie-Breaking ($s_i = -\text{len}(y_i)$) Is Pathological
When outcome rewards collapse, one might attempt to break ties by favoring shorter sequences ($s_i = -\text{len}(y_i)$). However, as demonstrated empirically across $M=10$ software bug groups in Section 5.5, **a shorter trajectory is not necessarily a better trajectory**: if a rollout group contains a truncated failure or refusal string (e.g., `"No patch."`, 9 characters) alongside a complete 240-character SRI patch, naive length tie-breaking assigns the *maximum positive advantage* to the 9-character failure and a *negative advantage* to the valid patch, actively training the policy to emit empty or truncated outputs.

#### Zero-Leakage Localization Rubric ($S_{\text{loc}}$) from Observable Context Only (Addressing Reviewer Issue 3)
A critical methodological requirement in RLVR is that process rewards must never leak privileged benchmark labels (such as gold patch file paths) that would be unavailable at deployment time. In `RubricProcessRewardModel` ([`src/rlvr/prm.py#L46-L135`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/prm.py#L46-L135)), `extract_observable_files_from_context(observable_context)` extracts candidate file paths $\mathcal{F}_{\text{obs}}$ **strictly from the agent's observable prompt and tool-interaction history**:
1. **Inspected `view_file` Headers:** Any file path inspected by the agent during multi-turn exploration (`=== File: <path> ===`).
2. **Tool Call Arguments:** Any Python file path passed to `<tool_call>{"name": "view_file", "arguments": {"path": "<path>"}}</tool_call>`.
3. **Public Issue / Traceback Mentions:** Any relative `.py` path explicitly mentioned in the public GitHub issue description or stack trace.

Given $\mathcal{F}_{\text{obs}}$ extracted from `observable_context` (with **zero access to the ground-truth benchmark patch**), the localization rubric $S_{\text{loc}}(y_i) \in \{0.0, 0.2, 0.5, 1.0\}$ scores whether the rollout's `<<<<<<< SEARCH: <file_path>` block targets a file that the agent actually inspected or that appeared in the issue traceback ($1.0$), mentions an observable file without editing it ($0.5$), or hallucinates an uninspected file path ($0.2$). When no `.py` paths appear in `observable_context` ($\mathcal{F}_{\text{obs}} = \emptyset$), $S_{\text{loc}}$ defaults to a neutral score ($0.5$).

#### Exact Two-Tier AVSPO Estimator Implemented in `src/rlvr/avspo.py`
Whenever group outcome variance collapses ($\sigma_{\mathbf{R}} < \epsilon_{\text{var}}$, $\epsilon_{\text{var}} = 10^{-6}$), `AVSPOAdvantageEstimator` ([`src/rlvr/avspo.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/avspo.py)) executes a two-tier fallback:

1. **Tier 1 (Zero-Leakage Rubric PRM Substitution):** Each rollout $y_i$ is scored by `RubricProcessRewardModel.evaluate_trajectory(y_i, observable_context=obs_ctx)` ([`src/rlvr/prm.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/prm.py)):
   $$P_i = 0.40\,S_{\text{fmt}}(y_i) + 0.35\,S_{\text{loc}}(y_i \mid \mathcal{F}_{\text{obs}}) + 0.25\,S_{\text{disc}}(y_i) \in [0, 1]$$
   where $S_{\text{fmt}} \in \{0.0, 0.3, 0.5, 1.0\}$ verifies parseable SRI block structure, $S_{\text{loc}}$ verifies consistency with observable inspected files $\mathcal{F}_{\text{obs}}$, and $S_{\text{disc}} \in \{0.2, 1.0\}$ penalizes repetitive command loops. If $\sigma_{\mathbf{P}} > \epsilon_{\text{var}}$, Tier 1 substitutes $P_i$ into both numerator and denominator:
   $$\hat{A}_i^{\text{Tier-1}} = \frac{P_i - \mu_{\mathbf{P}}}{\sigma_{\mathbf{P}} + \epsilon}$$

2. **Tier 2 (Structural-Completeness-Gated Virtual Reward Perturbation):** When both $\sigma_{\mathbf{R}} < \epsilon_{\text{var}}$ and $\sigma_{\mathbf{P}} \le \epsilon_{\text{var}}$, Tier 2 ranks rollouts using `compute_structural_tie_breaker` ([`src/rlvr/avspo.py#L127-L157`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/avspo.py#L127-L157)):
   $$s_i^{\text{struct}} = \mathbb{I}\!\left[\text{valid\_sri}(y_i)\right] + 0.25\,\mathbb{I}\!\left[\text{closed\_thought}(y_i)\right] - 1.0\,\mathbb{I}\!\left[\text{len}(y_i) < 32\right] - 10^{-4}\,\text{len}(y_i)$$
   Sorting the zero-mean linear perturbation vector $\boldsymbol{\Delta} = \text{linspace}(-\delta, +\delta, G)$ ($\delta = 0.15$) by permutation $\pi = \text{argsort}(\mathbf{s}^{\text{struct}})$ perturbs the reward vector itself:
   $$\tilde{R}_i = R_i + \Delta_{\pi^{-1}(i)}, \qquad \hat{A}_i^{\text{Tier-2}} = \frac{\tilde{R}_i - \mu_{\tilde{\mathbf{R}}}}{\sigma_{\tilde{\mathbf{R}}} + \epsilon} = \frac{\Delta_{\pi^{-1}(i)}}{\sigma_{\boldsymbol{\Delta}} + \epsilon} \neq 0$$

---

## 4. Experimental Setup, Artifact Auditability, and Reproducibility Protocol

### 4.1 Hardware and Software Environment
* **Hardware:** Single Apple M2 Max workstation (12-core CPU, 38-core GPU, **64.0 GB Unified Memory** / `68,719,476,736` bytes).
* **Software Stack:** macOS `26.6.2` (`arm64`), Python `3.10.19` (`uv`, locked via [`uv.lock`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/uv.lock)), PyTorch `2.14.0` (`mps` backend, `torch.bfloat16`), HuggingFace `transformers 5.17.0`, `datasets 5.0.1`, `pytest 9.1.1`.

### 4.2 Evaluation Stages, Sample Sizes, and Statistical Precision (Addressing Reviewer Issues 1 & 5)
Table 3 defines the exact model, sample size $N$, and statistical framing of every experimental stage, explicitly distinguishing controlled `ToyLM` multi-seed mechanisms from live `gemma-2-2b-it` training telemetry and paired $N=4$ repository sandbox evaluations.

| Stage ID | Experimental Stage Name | Model / Testbed | Sample Size & Provenance | Measured Outcome Types | Statistical Framing & Scope Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Zero-Shot Syntactic Baseline | `google/gemma-2-2b-it` (Frozen Base) | **$N = 25$** (`princeton-nlp/SWE-bench_Lite` test indices `0..24`) | Syntactic SRI Validity (`%`), Single-Turn Latency (`s`), Zero-Shot Resolution (`%`) | Exact 95% Clopper-Pearson CI on `21/25` is `[63.9%, 95.5%]`; on `0/25` is `[0.0%, 13.7%]`. |
| **Stage 2** | Multi-Turn SFT Curriculum Training | `google/gemma-2-2b-it` + `DR-LoRA` (`182` modules) | **$N = 40$ trajectories** (`3` epochs = `120` trajectory updates, [`data/sft_trajectories.jsonl`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/data/sft_trajectories.jsonl)) | Masked Cross-Entropy Loss, Active Rank Distribution ($r_m$), Update Norm $\|\Delta W\|_F$ | Optimization & capacity distribution telemetry; audited in [`results/ablation_and_audit_log.json`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/results/ablation_and_audit_log.json). |
| **Stage 3** | Controlled Multi-Seed Component & Estimator Benchmarks | Deterministic 4-Layer / 2-Layer `ToyLM` ($d=128$) | **$3$ Seeds (`42, 43, 44`)**; AVSPO tested across **$M=10$ distinct bug task groups** ($30$ group optimizations/estimator) | Cross-Entropy Loss, Erroneous vs. Valid Token NLL, Policy Gradient $\|\nabla_\theta \mathcal{L}\|_2$, **Executable Python Unit-Test Resolution (`k/30`)** | Controlled mechanistic testbed with mean $\pm$ SD across seeds and executable unit-test verification of selected patches. |
| **Stage 4** | Live RLVR Post-Training Telemetry | `google/gemma-2-2b-it` + `DR-LoRA` + `GRPOTrainer` | **$3$ RLVR Steps** $\times$ **$G = 4$ rollouts/step** (`12` generated rollouts on `mps`) | Composite Reward, Surrogate Loss, KL Divergence ($D_{\text{KL}}$), Advantage Collapse Rate (ACR), $T=0.7$ SRI Stability | Verifies critic-free memory feasibility (`12.60 GB` peak MPS RAM) and `0.00` ACR on `gemma-2-2b-it`. |
| **Stage 5** | Paired End-to-End Sandbox Verification & Component-Removal Audit | `google/gemma-2-2b-it` across **7 Paired Configurations (`C0`, `C1`, `C2`, `C3`, `C4a`, `C4b`, `C5`)** | **$N = 4$ official `SWE-bench_Lite` instances** (`django-10914`, `django-10924`, `astropy-12907`, `astropy-14182`) in isolated git worktrees | Per-Instance Verdict, Syntactic SRI Validity, Clean Patch Application, `FAIL_TO_PASS`, `PASS_TO_PASS`, Dual-Suite Resolution | **Paired descriptive evaluation** on the identical 4 tasks (`[6.8%, 93.2%]` 95% CI at `2/4`); establishes local pipeline feasibility and component necessity. |

### 4.3 Independent Reproducibility Audit and Exact CLI Commands (Addressing Reviewer Issue 4)
To guarantee that an external reviewer cloning `https://github.com/vishnu-anirudh/gemma-swe-agent` (with pinned dependency lockfile [`uv.lock`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/uv.lock)) knows exactly which artifacts are immediately executable vs. which require regenerating the `166.2 MB` (`166,245,395` bytes) SFT checkpoint (excluded from Git push due to GitHub's 100 MB file limit), Table 3B documents the exact CLI commands, tracked file paths, and runtime requirements.

#### Table 3B: Reproducibility Matrix — Immediate Verification vs. Full Checkpoint Regeneration
| Verification Tier | Required Local Files | Exact CLI Command (from Repository Root) | Expected Runtime (Apple M2 Max) | Verified Outputs & Tracked Artifacts |
| :--- | :--- | :--- | :---: | :--- |
| **Tier A1: Full Unit & Integration Test Suite (34 Tests)** | Tracked Git repo only *(no `.pt` checkpoint required)* | `PYTHONPATH=. uv run pytest -v` | **~1.5 s** | Verifies `SRIFormatter` (6 tiers), `DRLoRAManager`, `StepLevelErrorMasker`, `AVSPOAdvantageEstimator`, `RubricProcessRewardModel`, `MultiTurnAgent`, and `ParallelDistillRefineEngine`. |
| **Tier A2: Controlled Multi-Seed Ablations & Statistical Audit** | Tracked Git repo only *(if `./checkpoints/sft_dr_lora/adapter_model.pt` is present locally, also re-audits its 182 tensors)* | `PYTHONPATH=. uv run python scripts/run_ablations_and_audit.py` | **~18.2 s** | Regenerates [`results/ablation_and_audit_log.json`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/results/ablation_and_audit_log.json) (`45,238` bytes): Clopper-Pearson 95% CIs, 3-seed PEFT ablation, 3-seed masking ablation, $M=10 \times 3$-seed AVSPO unit-test benchmark, and `C0`–`C5` audit. |
| **Tier B1: Regenerate DR-LoRA SFT Checkpoint (`166.2 MB`)** | Tracked Git repo + HuggingFace `google/gemma-2-2b-it` cache | `PYTHONPATH=. uv run python -m src.train --mode sft --config configs/sft_dr_lora.yaml --epochs 3` | **~14 min** | Writes `./checkpoints/sft_dr_lora/adapter_model.pt` (`182` adapted modules, `10,627,072` active parameters; reference SHA-256: `02598ff756435f92536b23a7c094a27c7d55658c8805202e0c8f798d38b55b93`). |
| **Tier B2: Live Critic-Free GRPO-AVSPO Post-Training** | Tracked Git repo + Tier B1 checkpoint | `PYTHONPATH=. uv run python -m src.train --mode rlvr --config configs/rlvr_grpo.yaml --steps 3` | **~6 min** | Executes 3 critic-free GRPO-AVSPO steps ($G=4$) via [`src/rlvr/grpo.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/rlvr/grpo.py) and saves `./checkpoints/rlvr_grpo/adapter_model.pt`. |
| **Tier B3: End-to-End Multi-Turn Sandbox & PDR Evaluation** | Tracked Git repo + Tier B1/B2 checkpoint + cached `SWE-bench_Lite` git repos | `PYTHONPATH=. uv run python -m src.train --mode eval --config configs/sft_dr_lora.yaml --checkpoint ./checkpoints/sft_dr_lora/adapter_model.pt --max-samples 4` | **~1.0–1.9 min / task** | Invokes [`MultiTurnAgent`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/agent/agent_loop.py) and [`ParallelDistillRefineEngine`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/evaluation/parallel_distill_refine.py) in isolated git worktrees and writes [`results/post_train_eval_report.json`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/results/post_train_eval_report.json). |

---

## 5. Empirical Results and Controlled Component Ablations

### 5.1 Paired Component-Removal Matrix and Per-Instance Outcome Audit (Addressing RQ1, RQ4, & Reviewer Issue 2)
In our previous manuscript draft, Configuration `C4` and Configuration `C5` both reported `2/4` resolved tasks because `C4` already included the 2nd-round test-guided failure distillation and refinement retry inside `MultiTurnAgent.run_session` ([`src/agent/agent_loop.py#L807-L920`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/agent/agent_loop.py#L807-L920)). To isolate each component's true downstream contribution without confounding 1-round synthesis, 3-step GRPO-AVSPO, and 2-round test-guided refinement (Reviewer Issue 2), Table 4A and Table 4B report **seven paired configurations (`C0`, `C1`, `C2`, `C3`, `C4a`, `C4b`, `C5`)** evaluated on the identical $N=4$ sandbox task set:
* **`C4a` (Multi-Turn + DR-LoRA + Masked SFT, 1-Round Single-Pass Synthesis, No Test-Guided Refinement):** Evaluates the first synthesized patch ([`src/agent/agent_loop.py#L747-L775`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/agent/agent_loop.py#L747-L775)) directly against the dual test suite without any test-feedback retry.
* **`C4b` (Multi-Turn + DR-LoRA + Masked SFT + 3-Step GRPO-AVSPO, 1-Round Single-Pass Synthesis, No Test-Guided Refinement):** Evaluates the checkpoint after 3 steps of local GRPO-AVSPO in 1-round single-pass synthesis.
* **`C5` (Full Pipeline: `C4b` + 2-Round Test-Guided Failure Distillation & PDR $K=4$):** Executes multi-candidate temperature sampling ($K=4$) and 2nd-round failure-distilled patch refinement ([`src/evaluation/parallel_distill_refine.py`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/evaluation/parallel_distill_refine.py) & [`src/agent/agent_loop.py#L807-L920`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/src/agent/agent_loop.py#L807-L920)) when Round 1 fails the unit-test harness.

#### Table 4A: Paired Component-Removal / Cumulative Addition Matrix ($N=4$ Held-Out Sandbox Tasks & $N=25$ Zero-Shot Slice)
| Config ID & Isolated Pipeline Stage | Eval Split ($N$) | Metric 1: Syntactic SRI Valid (`k/N`, %, 95% CI) | Metric 2: Patch Applied Cleanly (`k/N`, %, 95% CI) | Metric 3: `FAIL_TO_PASS` Bug Fixed (`k/N`, %, 95% CI) | Metric 4: `PASS_TO_PASS` No Regression (`k/N`, %, 95% CI) | Metric 5: Full Dual-Suite Resolved (`k/N`, %, 95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0-Lite`: Zero-Shot Base** *(1-round, no tools, no SFT)* | Lite Slice ($N=25$) | `21/25` (**84.0%**, `[63.9%, 95.5%]`) | `0/25` (0.0%, `[0.0%, 13.7%]`) | `0/25` (0.0%, `[0.0%, 13.7%]`) | — | **`0/25` (0.0%, `[0.0%, 13.7%]`)** |
| **`C0`: Zero-Shot Base** *(1-round, no tools, no SFT)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | `0/4` (0.0%, `[0.0%, 60.2%]`) | `0/4` (0.0%, `[0.0%, 60.2%]`) | `0/4` (0.0%, `[0.0%, 60.2%]`) | **`0/4` (0.0%, `[0.0%, 60.2%]`)** |
| **`C1`: + Multi-Turn Localization** *(Base model + `view_file`/`grep`, 1-round)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | `0/4` (0.0%, `[0.0%, 60.2%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | **`0/4` (0.0%, `[0.0%, 60.2%]`)** |
| **`C2`: + Unmasked Trajectory SFT** *(`mask_error_actions=False`, Uniform $r=8$, 1-round)* | Sandbox ($N=4$) | `3/4` (75.0%, `[19.4%, 99.4%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | **`1/4` (25.0%, `[0.6%, 80.6%]`)** |
| **`C3`: + Step-Level Error Masking** *(`mask_error_actions=True`, Uniform $r=8$, 1-round)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | `2/4` (50.0%, `[6.8%, 93.2%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | `1/4` (25.0%, `[0.6%, 80.6%]`) | **`1/4` (25.0%, `[0.6%, 80.6%]`)** |
| **`C4a`: + Dynamic Rank LoRA (`DR-LoRA`)** *($r_0=8, \gamma=0.4$, **1-Round Single-Pass, No PDR**)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | **`3/4` (75.0%, `[19.4%, 99.4%]`)** | `1/4` (25.0%, `[0.6%, 80.6%]`) | `2/4` (50.0%, `[6.8%, 93.2%]`) | **`1/4` (25.0%, `[0.6%, 80.6%]`)** |
| **`C4b`: + 3-Step GRPO-AVSPO** *(**1-Round Single-Pass, No PDR**)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | **`3/4` (75.0%, `[19.4%, 99.4%]`)** | `1/4` (25.0%, `[0.6%, 80.6%]`) | `2/4` (50.0%, `[6.8%, 93.2%]`) | **`1/4` (25.0%, `[0.6%, 80.6%]`)** |
| **`C5`: + 2-Round Test-Guided Failure Distillation & PDR ($K=4$)** *(Full Pipeline)* | Sandbox ($N=4$) | `4/4` (**100.0%**, `[39.8%, 100.0%]`) | **`3/4` (75.0%, `[19.4%, 99.4%]`)** | **`2/4` (50.0%, `[6.8%, 93.2%]`)** | **`3/4` (75.0%, `[19.4%, 99.4%]`)** | **`2/4` (50.0%, `[6.8%, 93.2%]`)** |

#### Table 4B: Paired Per-Instance Verdict Progression Across All 7 Configurations (`C0`–`C5`)
| Configuration ID | Task 1: `django__django-10914` (`global_settings.py:304`) | Task 2: `django__django-10924` (`fields/__init__.py:1709`) | Task 3: `astropy__astropy-12907` (`separable.py:242`) | Task 4: `astropy__astropy-14182` (`rst.py:57`) | Paired Resolved Count (`k/4`) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`C0` (Zero-Shot Base)** | **FAILED** *(No patch applied; hallucinated search lines)* | **FAILED** *(No patch applied; hallucinated search lines)* | **FAILED** *(No patch applied; hallucinated search lines)* | **FAILED** *(No patch applied; hallucinated search lines)* | **`0/4` (0.0%)** |
| **`C1` (+ Multi-Turn Base)** | **PARTIAL** *(Applied, but sets wrong permission value; `FAIL_TO_PASS` fails)* | **FAILED** *(Unaligned search block; fails to apply)* | **FAILED** *(Unaligned search block; fails to apply)* | **FAILED** *(Single-block mismatch; fails to apply)* | **`0/4` (0.0%)** |
| **`C2` (+ Unmasked SFT, 1-Round)** | **RESOLVED** *(`0o644` applied; passes `1/1` F2P & `12/12` P2P)* | **FAILED** *(Imitates intermediate error tool call; no valid apply)* | **FAILED** *(Search block mismatch; fails to apply)* | **FAILED** *(Malformed SRI block syntax)* | **`1/4` (25.0%)** |
| **`C3` (+ Error Masking, Uniform $r=8$, 1-Round)** | **RESOLVED** *(`0o644` applied; passes `1/1` F2P & `12/12` P2P)* | **FAILED** *(Applies `'path': self.path(),` without `callable()` guard; fails string-path `PASS_TO_PASS`)* | **FAILED** *(Indentation mismatch in `_cstack`; fails to apply)* | **FAILED** *(Valid SRI syntax, but single-hunk mismatch)* | **`1/4` (25.0%)** |
| **`C4a` (+ DR-LoRA, 1-Round Single-Pass)** | **RESOLVED** *(`0o644` applied; passes `1/1` F2P & `12/12` P2P)* | **FAILED** *(Applies `'path': self.path(),` in Round 1; fails string-path `PASS_TO_PASS` with `TypeError`)* | **PARTIAL (0 Regressions)** *(Applies cleanly; passes `19/19` `PASS_TO_PASS`, fails `FAIL_TO_PASS`)* | **FAILED** *(Valid SRI syntax, single-hunk mismatch)* | **`1/4` (25.0%)** |
| **`C4b` (+ 3-Step GRPO-AVSPO, 1-Round Single-Pass)** | **RESOLVED** *(`0o644` applied; passes `1/1` F2P & `12/12` P2P)* | **FAILED** *(Still emits unguarded `'path': self.path(),` in Round 1; fails string-path `PASS_TO_PASS`)* | **PARTIAL (0 Regressions)** *(Applies cleanly; passes `19/19` `PASS_TO_PASS`, fails `FAIL_TO_PASS`)* | **FAILED** *(Valid SRI syntax, single-hunk mismatch)* | **`1/4` (25.0%)** |
| **`C5` (+ 2-Round Test-Guided Distillation & PDR $K=4$)** | **RESOLVED** *(Verified Round-1 winner selected)* | **RESOLVED on Round 2** *(Round-1 `TypeError: 'str' object is not callable` distilled $\to$ emits `'path': self.path() if callable(self.path) else self.path,`)* | **PARTIAL (0 Regressions)** *(Applies cleanly; passes `19/19` `PASS_TO_PASS`, fails `FAIL_TO_PASS`)* | **FAILED** *(Requires coordinated edits across `RST.__init__` and `RST.write`)* | **`2/4` (50.0%)** |

#### Four Candid Empirical Takeaways from the Separated Ablation Matrix (`C0`–`C5`)
1. **What Step-Level Error Masking (`C2 -> C3`) Actually Changes:** In 1-round synthesis, both `C2` (Unmasked SFT) and `C3` (Masked SFT) resolve `1/4` tasks (`django-10914`). However, `C3` restores `4/4` (`100%`) syntactic SRI compliance and enables clean patch application on `django-10924` (`1/4 -> 2/4` patches applied), whereas `C2` imitates intermediate erroneous tool calls on `django-10924`.
2. **What DR-LoRA (`C3 -> C4a`) Actually Changes:** In 1-round single-pass synthesis, replacing Uniform LoRA ($r=8$, `C3`) with DR-LoRA (`C4a`) increases **clean patch application from `2/4` (`50.0%`) to `3/4` (`75.0%`)** and **`PASS_TO_PASS` non-regression from `1/4` (`25.0%`) to `2/4` (`50.0%`)** by enabling a clean, non-regressing edit on `astropy__astropy-12907` (`19/19` regression tests pass). However, `C4a` alone does **not** increase 1-round full resolution (`1/4` in both `C3` and `C4a`), because on `django-10924` the 1-round patch writes `'path': self.path(),` without guarding for standard string paths.
3. **What 3-Step Local GRPO-AVSPO (`C4a -> C4b`) Does—and Does Not—Achieve:** In 1-round single-pass synthesis on the $N=4$ sandbox slice, `C4b` achieves the **exact same task-level resolution as `C4a` (`1/4` = `25.0%` resolved, `3/4` patch applied)**. We state plainly that **3 steps of local GRPO-AVSPO (`12` rollouts) are insufficient to teach the 2.61B model new repository bug-fixing semantics**. Instead, the empirical role of Stage 4 (`C4b`) in this study is twofold: (a) validating that critic-free GRPO-AVSPO executes within **`12.60 GB` peak MPS memory** at **`0.00` Advantage Collapse Rate**, and (b) improving multi-sample syntactic SRI formatting stability at sampling temperature $T=0.7$ from **`13/16` (`81.3%`) in `C4a` to `16/16` (`100.0%`) in `C4b`**, which provides cleaner candidate pools for multi-sample test-time scaling (`C5`).
4. **Why 2-Round Test-Guided Failure Distillation & PDR (`C4b -> C5`) Drives the Gain from `1/4` (`25.0%`) to `2/4` (`50.0%`):** On `django__django-10924`, Round 1's patch (`'path': self.path(),`) fixes the new callable-path test (`FAIL_TO_PASS`), but breaks existing string-path `FilePathField` instances in `PASS_TO_PASS` with `TypeError: 'str' object is not callable`. When `C5` distills this exact traceback via `distill_failure_summary` and prompts Round 2 refinement, the model synthesizes the guarded ternary expression `'path': self.path() if callable(self.path) else self.path,`, passing both `FAIL_TO_PASS` (`1/1`) and `PASS_TO_PASS` (`28/28`).

---

### 5.2 Per-Instance Sandbox Execution Audit and Wall-Clock Latency Breakdown
Table 5 summarizes the final verified state (`C5`) on the $N=4$ sandbox slice, and Table 6 decomposes wall-clock runtime on Apple M2 Max (`mps`).

| Instance ID | Target File & Fault Location | Patch Applied | `FAIL_TO_PASS` Suite | `PASS_TO_PASS` Suite | Final Verdict & Root-Cause Audit (`C5`) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `django__django-10914` | `django/conf/global_settings.py:304` | **Yes** (Tier-1 Exact) | **PASSED** (`1/1`) | **PASSED** (`12/12`) | **RESOLVED (Round 1):** Updates `FILE_UPLOAD_PERMISSIONS = None` $\to$ `0o644`. |
| `django__django-10924` | `django/db/models/fields/__init__.py:1709` | **Yes** (Tier-6 Fuzzy + Indent Delta) | **PASSED** (`1/1`) | **PASSED** (`28/28`) | **RESOLVED (Round 2 via Traceback Distillation):** Round 1 emits `self.path()`; Round 2 refines to `self.path() if callable(self.path) else self.path`. |
| `astropy__astropy-12907` | `astropy/modeling/separable.py:242` | **Yes** (Tier-4 Line Match) | **FAILED** (`0/1`) | **PASSED** (`19/19`) | **PARTIAL (No Regression):** Applies cleanly and passes all 19 regression tests, but sets `cright = 1` instead of slice `right` for nested `CompoundModel`. |
| `astropy__astropy-14182` | `astropy/io/ascii/rst.py:57` | **No** (Multi-hunk mismatch) | **FAILED** (`0/1`) | — | **FAILED:** Requires coordinated edits across `RST.__init__` and `RST.write`; 2B model emits single-block edit only. |

#### Table 6: Wall-Clock Latency Decomposition on Apple M2 Max (64 GB Unified Memory)
| Execution Sub-Phase | Stage 1: Zero-Shot Single-Turn ($N=25$) | Stage 5 (`C4a`/`C4b`): 1-Round Multi-Turn Agent | Stage 5 (`C5`): 2-Round Refinement / PDR (`K=4`) | Measurement Boundary |
| :--- | :---: | :---: | :---: | :--- |
| **1. Model Autoregressive Generation (`mps`, `bfloat16`)** | **3.80 s** $\pm$ 0.42 s | **21.40 s** $\pm$ 3.85 s | **68.20 s** $\pm$ 6.10 s | Pure GPU/MPS forward passes and token decoding only. |
| **2. Git Worktree Checkout & 6-Tier SRI Patch Application** | 0.00 s *(not invoked)* | **1.85 s** $\pm$ 0.30 s | **4.90 s** $\pm$ 0.65 s | Detached git worktree creation, `test_patch` apply, and SRI matching. |
| **3. Dual-Suite Sandbox Test Execution (`pytest` / Django)** | 0.00 s *(not invoked)* | **36.50 s** $\pm$ 4.10 s | **42.10 s** $\pm$ 5.20 s | Subprocess execution of `FAIL_TO_PASS` and `PASS_TO_PASS` suites. |
| **Total End-to-End Wall-Clock Time per Instance** | **3.80 s** *(generation only)* | **59.75 s** *(1-round verification)* | **115.20 s** *(4-way tournament)* | Complete wall-clock time from issue prompt to verified test verdict. |

---

### 5.3 Ablation 1: DR-LoRA Checkpoint Audit (`gemma-2-2b-it`) and 3-Seed Controlled PEFT Comparison (Addressing RQ2)
Table 7 reports the exact layer-type parameter and rank audit extracted from the trained `google/gemma-2-2b-it` checkpoint (`./checkpoints/sft_dr_lora/adapter_model.pt`, SHA-256 `02598ff7...`, `182` adapted modules across 26 layers), and Table 8 reports the 3-seed controlled benchmark (`seeds = [42, 43, 44]`, `ToyLM`) isolating dynamic rank reallocation and EMA step-transition smoothing.

#### Table 7: Post-SFT Rank and Spectral Audit of `google/gemma-2-2b-it` (`./checkpoints/sft_dr_lora/adapter_model.pt`)
| Projection Module Type | Adapted Layers ($L$) | Initial Rank ($r_0$) | Post-SFT Mean Active Rank | Min / Max Active Rank | Mean Update Norm $\|\Delta W\|_F$ | Top-$r$ Singular Energy Share | Active Trainable Params / Layer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v_proj`** (Attention Value) | 26 | 8 | **8.88** | `7` / `13` | $0.0216 \pm 0.0008$ | **98.57%** | 29,568 |
| **`up_proj`** (MLP Up-Projection) | 26 | 8 | **8.88** | `7` / `12` | $0.0641 \pm 0.0019$ | **99.02%** | 102,350 |
| **`o_proj`** (Attention Output) | 26 | 8 | **8.69** | `7` / `12` | $0.0322 \pm 0.0009$ | **96.66%** | 37,828 |
| **`gate_proj`** (MLP GeGLU Gate) | 26 | 8 | **8.62** | `7` / `15` | $0.0623 \pm 0.0013$ | **96.17%** | 99,249 |
| **`down_proj`** (MLP Down-Projection) | 26 | 8 | **7.81** | `6` / `9` | $0.0322 \pm 0.0012$ | **96.60%** | 89,944 |
| **`q_proj`** (Attention Query) | 26 | 8 | **6.50** | `5` / `14` | $0.0259 \pm 0.0012$ | 78.02% | 28,288 |
| **`k_proj`** (Attention Key) | 26 | 8 | **6.46** | `5` / `15` | $0.0182 \pm 0.0008$ | 77.47% | 21,504 |
| **Global Total / Mean** | **182** | **8.00** | **7.98** | `5` / `15` | $0.0366 \pm 0.0178$ | **91.79%** | **10,627,072 total active** |

#### Table 8: Controlled 3-Seed PEFT Ablation (`seeds = [42, 43, 44]`, `ToyLM`, Audited in `results/ablation_and_audit_log.json`)
| PEFT Variant | Active Rank Budget | Initial Loss | Final Loss (Mean $\pm$ SD) | Seed-Specific Final Losses (`s42 / s43 / s44`) | Next-Token Exact-Match Acc | Cross-Module Rank SD ($\sigma_r$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform LoRA (Fixed $r=8$)** | 8.00 | 5.6955 | $0.8808 \pm 0.1341$ | `0.6929 / 0.9942 / 0.9553` | 82.75% | 0.00 |
| **DR-LoRA without Smoothing ($\gamma=1.0$)** | 7.87 | 5.6955 | $0.8139 \pm 0.1126$ | `0.6933 / 0.7819 / 0.9665` | **85.61%** | 4.66 |
| **DR-LoRA with EMA Smoothing ($\gamma=0.4$)** | 8.08 | 5.6955 | **$0.8149 \pm 0.2176$** | `0.5567 / 0.7873 / 1.1008` | **83.66%** | **3.94** (-15.5% rank variance) |

---

### 5.4 Ablation 2: Step-Level Error Masking vs. Unmasked SFT (`seeds = [42, 43, 44]`, Addressing RQ2)
Table 9 reports the 3-seed controlled ablation comparing `StepLevelErrorMasker(mask_error_actions=True)` against unmasked SFT (`mask_error_actions=False`).

| Supervision Strategy | Supervised Tokens / Total | Erroneous Action Token NLL (Mean $\pm$ SD, $\uparrow$ better) | Erroneous Action Token Prob ($\downarrow$ better) | Valid `<final_patch>` Token NLL (Mean $\pm$ SD, $\downarrow$ better) | Valid `<final_patch>` Token Prob ($\uparrow$ better) | Seed-Specific Erroneous NLL (`s42 / s43 / s44`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmasked SFT (`mask_error_actions=False`)** | `375 / 609` (100% assistant) | $1.7037 \pm 0.0285$ | $18.21\%$ | $1.3950 \pm 0.0082$ | $24.79\%$ | `1.6641 / 1.7176 / 1.7294` |
| **Step-Level Error Masked SFT (`mask_error_actions=True`)** | `285 / 609` (**24.0% error tokens masked**) | **$5.5455 \pm 0.1990$** | **$0.40\%$** (**45.7x suppression**) | **$1.2816 \pm 0.0027$** | **$27.76\%$** (+12.0% relative) | `5.3204 / 5.5029 / 5.8131` |

---

### 5.5 Ablation 3: Multi-Group Validation of 5 AVSPO Estimators with Zero-Leakage PRM and Executable Unit-Test Verification (Addressing RQ3 & Reviewer Issue 3)
To address Reviewer Issue 3—specifically (a) expanding evaluation beyond a single hand-crafted rollout group to a diverse multi-group benchmark, (b) computing localization process rewards ($S_{\text{loc}}$) strictly from observable context without gold-patch leakage, and (c) verifying whether post-RLVR Top-1 selected patches actually resolve executable unit tests—we evaluated **five advantage estimators** across **$M = 10$ distinct Python software bug task groups** (`T1`–`T10`, covering file permissions, callable path handling, zero-division guards, bounds checking, UTF-8 fallback decoding, empty-sequence mean, set discard, mutable default isolation, dictionary pop defaults, and config key normalization) and **3 random seeds (`42, 43, 44`)**, totaling **$30$ group optimizations ($120$ candidate rollouts) per estimator** ([`scripts/run_ablations_and_audit.py#L405-L678`](file:///Users/vishnuanirudh/code/kaggle/gemma_challenge/scripts/run_ablations_and_audit.py#L405-L678)).

In each task group $m \in \{1, \dots, 10\}$ under homogeneous binary outcome failure ($\mathbf{R} = [0, 0, 0, 0]^\top$), the agent's observable prompt contains `=== File: <inspected_file> ===` from `view_file`, and the four candidate rollouts are:
* $y_0$ (**Valid Target SRI Patch on Inspected File**): Valid `<thought>` + valid SRI block fixing the bug in `<inspected_file>`.
* $y_1$ (**Valid SRI on Uninspected Distractor File**): Valid `<thought>` + valid SRI block targeting an uninspected distractor file with a shorter path string.
* $y_2$ (**Truncated Incomplete Failure**): 9-character refusal/truncated string `"No patch."`.
* $y_3$ (**Unformatted Rambling Prose**): Unstructured explanation without SRI markers.

After 10 GRPO policy update steps per group, we select the Top-1 rollout $\hat{y} = \arg\max_{y_i} \pi_\theta(y_i)$, parse it with `SRIFormatter.parse`, apply it to the buggy Python module via `SRIFormatter.apply_to_content`, and execute the task's Python unit-test suite (`exec` + unit-test assertions) to measure **Downstream Executable Unit-Test Resolution (`k/10` per seed, `k/30` total)**.

#### Table 10: Multi-Group 3-Seed Comparison of 5 Advantage Estimators ($M=10$ Bug Task Groups $\times 3$ Seeds = $30$ Group Optimizations / Estimator)
| Advantage Estimator Variant | Step-1 Policy Grad Norm $\|\nabla_\theta \mathcal{L}\|_2$ (Mean $\pm$ SD) | Post-RLVR Truncated `"No patch."` Prob ($y_2$, $\downarrow$ better) | Post-RLVR Valid SRI Total Prob ($y_0 + y_1$, $\uparrow$ better) | Post-RLVR Exact Target Patch Prob ($y_0$, $\uparrow$ better) | Top-1 Valid SRI Selection Rate (`30` groups) | **Downstream Executable Unit-Test Resolution Rate (`k/30`, Mean $\pm$ SD; Seeds `s42/s43/s44`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Vanilla GRPO** (Shao et al., 2024) | **$0.0000 \pm 0.0000$** | $24.32\%$ | $50.55\%$ | $25.30\%$ | `13/30` (43.33%) | **`8/30` (`26.67%` $\pm$ `4.71%`; `3/10, 2/10, 3/10`)** |
| **2. AVSPO Tier-2 (Naive Length: $s_i = -\text{len}(y_i)$)** | $1.5995 \pm 0.0114$ | **$96.67\%$ (Pathological!)** | $3.33\%$ | $0.00\%$ | `1/30` (3.33%) | **`0/30` (`0.00%` $\pm$ `0.00%`; `0/10, 0/10, 0/10`)** |
| **3. AVSPO Tier-2 (Gaussian Noise: $\Delta_i \sim \mathcal{N}(0, \delta^2)$)** | $1.2095 \pm 0.0901$ | $46.67\%$ | $30.00\%$ | $6.67\%$ | `9/30` (30.00%) | **`2/30` (`6.67%` $\pm$ `9.43%`; `0/10, 2/10, 0/10`)** |
| **4. AVSPO Tier-2 (Structural-Gated: $s_i^{\text{struct}}$)** | $1.5611 \pm 0.0090$ | **$3.33\%$ (Trunc. Eliminated)** | **$86.67\%$** | $16.67\%$ *(picks shorter $y_1$)* | **`26/30` (86.67%)** | **`5/30` (`16.67%` $\pm$ `4.71%`; `1/10, 2/10, 2/10`)** |
| **5. AVSPO Tier-1 (Zero-Leakage Rubric PRM: $P_i$)** | **$1.2146 \pm 0.0070$** | **$0.00\%$ (Eliminated)** | **$96.67\%$** | **$66.67\%$** | **`29/30` (96.67%)** | **`20/30` (`66.67%` $\pm$ `4.71%`; `6/10, 7/10, 7/10`)** |

#### Three Critical Takeaways from the 10-Task Multi-Group Benchmark (Table 10)
1. **Naive Length Tie-Breaking Destroys Downstream Task Resolution (`26.67% -> 0.00%`):** Across 30 group optimizations, Naive Length Tie-Breaking ($s_i = -\text{len}(y_i)$) produces large gradients ($\|\nabla_\theta \mathcal{L}\|_2 = 1.5995$), but assigns the maximum positive advantage ($+1.3416$) to the 9-character string `"No patch."` ($y_2$). This drives post-update truncated output probability to **`96.67%`** and reduces downstream executable unit-test resolution from `8/30` (`26.67%` under zero-gradient Vanilla GRPO) to **`0/30` (`0.00%`)**.
2. **Tier-2 Structural-Completeness Gating Solves Truncation Collapse (`96.67% -> 3.33%`), But Cannot Disambiguate Target Files Alone:** Gating Tier-2 virtual perturbations by $s_i^{\text{struct}}$ reduces truncated output probability to **`3.33%`** and raises valid SRI selection to **`26/30` (`86.67%`)**. However, because Tier 2 does not inspect which file was viewed in the prompt, its secondary conciseness term ($-10^{-4}\text{len}(y_i)$) favors the shorter distractor-file SRI patch $y_1$ over $y_0$, resulting in `5/30` (`16.67%`) unit-test resolution. Thus, Tier 2 is strictly a **syntactic survival fallback** for when process scores tie ($\sigma_{\mathbf{P}} = 0$).
3. **Tier-1 Zero-Leakage Rubric PRM Achieves `66.67%` (`20/30`) Executable Unit-Test Resolution (+40.0 Percentage Points over Vanilla GRPO):** By extracting inspected file headers (`=== File: <path> ===`) strictly from the agent's observable prompt (`observable_context`) without touching gold patches, Tier-1 Rubric PRM assigns the highest advantage ($+1.4776$) to $y_0$, achieving **`0.00%` truncated probability**, **`29/30` (`96.67%`) valid SRI selection**, and **`20/30` (`66.67%` $\pm$ `4.71%`) downstream unit-test resolution**.

#### Table 11: Live RLVR Execution Telemetry on `google/gemma-2-2b-it` (`mps`, Stage 4)
| RLVR Step | Group Size ($G$) | Mean Composite Reward | Surrogate Policy Loss | Mean KL Divergence ($D_{\text{KL}}$) | Advantage Collapse Detected | Rolling Advantage Collapse Rate (ACR) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1** | 4 | `0.8125` | `-0.0312` | `0.0018` | `False` | `0.00` |
| **Step 2** | 4 | `0.7650` | `-0.0184` | `0.0031` | `False` | `0.00` |
| **Step 3** | 4 | `0.7775` | `-0.0241` | `0.0038` | `False` | `0.00` |
| **3-Step Mean** | **4** | **`0.7850`** | **`-0.0246`** | **`0.0029`** | **`0/3` collapsed** | **`0.00`** |

---

## 6. Discussion, Epistemic Audit, Limitations, and Conclusion

### 6.1 Epistemic Audit: Separating Literature, Empirical Measurements, Hypotheses, and Future Work
Table 12 classifies every central statement in this study to ensure zero conflation between intermediate metrics and end-to-end resolution:

| Statement / Proposition | Epistemic Classification | Exact Scope, Denominator, & Verification Source |
| :--- | :--- | :--- |
| Enforcing the SRI grammar on frozen `gemma-2-2b-it` yields `21/25` (`84.0%`, 95% CI: `[63.9%, 95.5%]`) syntactically parseable patches at `3.80 s/instance` generation time, but `0/25` (`0.0%`, 95% CI: `[0.0%, 13.7%]`) resolved issues in zero-shot mode. | **(B) Verified Empirical Measurement** | Stage 1 ($N=25$ SWE-bench Lite slice); demonstrates that syntactic compliance does not imply semantic bug resolution without repository context. |
| In `gemma-2-2b-it` SFT (`N=40` trajectories), DR-LoRA assigns higher mean active rank to `v_proj` (`8.88`) and MLP `up_proj` (`8.88`) / `gate_proj` (`8.62`) than `q_proj` (`6.50`) and `k_proj` (`6.46`). | **(B) Verified Empirical Measurement** | Stage 2 (`182` modules in `./checkpoints/sft_dr_lora/adapter_model.pt`, SHA-256 `02598ff7...`, Table 7 & `results/ablation_and_audit_log.json`). |
| Why value and MLP projections receive higher rank than query/key projections: syntactic code transformations require higher-rank feature mappings than attention routing. | **(C) Mechanistic Hypothesis** | Plausible interpretation consistent with AdaLoRA (Zhang et al., 2023), framed strictly as a hypothesis rather than a proven causal law. |
| Across $M=10$ software bug groups $\times 3$ seeds ($30$ group optimizations), naive length tie-breaking ($s_i = -\text{len}(y_i)$) reduces executable unit-test resolution to `0/30` (`0.00%`), whereas zero-leakage observable-context Rubric PRM substitution increases unit-test resolution to `20/30` (`66.67%` $\pm$ `4.71%`). | **(B) Verified Empirical Measurement** | Stage 3 (10-task $\times$ 3-seed controlled `ToyLM` benchmark across 5 estimators, Table 10 & `results/ablation_and_audit_log.json`). |
| On the paired sandbox slice ($N=4$), 1-round DR-LoRA SFT (`C4a`) improves clean patch application to `3/4` (`75.0%`) and resolves `1/4` (`25.0%`), 3-step GRPO-AVSPO (`C4b`) maintains `1/4` (`25.0%`) in 1-round synthesis while improving $T=0.7$ SRI stability (`81.3% -> 100.0%`), and 2-round test-guided failure distillation / PDR (`C5`) increases resolution to `2/4` (`50.0%`, 95% CI: `[6.8%, 93.2%]`). | **(B) Verified Empirical Measurement** | Stage 5 ($N=4$ paired held-out sandbox tasks with full `FAIL_TO_PASS` and `PASS_TO_PASS` verification, Table 4A & Table 4B). |
| Scaling this local pipeline to the full 300-instance SWE-bench Lite or 500-instance SWE-bench Verified suites will maintain a 50% resolution rate. | **(D) Unverified Extrapolation / Future Work** | Explicitly **not claimed**; $N=4$ carries a 95% exact binomial confidence interval of `[6.8%, 93.2%]` and establishes local architectural feasibility only. |

---

### 6.2 Limitations and Threats to Validity
1. **Small End-to-End Sandbox Sample Size ($N=4$):** While Stage 1 evaluated $N=25$ instances for syntactic compliance, Stage 2 trained on $N=40$ multi-turn trajectories, and Stage 3 evaluated $M=10$ bug groups across 3 seeds ($120$ rollouts), full git-worktree dual-suite unit-test verification (Stage 5) was restricted to $N=4$ instances (`2` Django, `2` Astropy) due to local repository environment overhead on a single workstation. At $N=4$, an observed resolution rate of `2/4` (`50.0%`) has a 95% exact Clopper-Pearson confidence interval of **`[6.8%, 93.2%]`**. Accordingly, our end-to-end numbers must be interpreted strictly as a **paired local-hardware feasibility demonstration** on Apple Silicon, not as a claim of competitive standing against larger agents evaluated on the full 300-instance SWE-bench Lite suite.
2. **Short Horizon of Live RLVR Post-Training (`C4b`, 3 Steps):** On live `gemma-2-2b-it` (`mps`), GRPO-AVSPO was executed for 3 steps ($G=4$, 12 rollouts) to verify critic-free unified-memory bounds (`12.60 GB`) and zero advantage collapse (`ACR = 0.00`). As shown transparently in Table 4A/4B, 3 steps of RLVR do not resolve new tasks in 1-round greedy synthesis (`C4b = 1/4` vs. `C4a = 1/4`); the gain to `2/4` in `C5` comes from 2nd-round test-guided failure distillation and refinement. Multi-epoch RLVR training across hundreds of live repository sandboxes remains future work.
3. **Single-Block vs. Multi-Method Edit Horizon in Sub-3B Models:** As shown by `astropy__astropy-14182` (Table 5), `gemma-2-2b-it` succeeds on localized single-block edits (`django-10914`, `django-10924`) and applies single-block changes cleanly without regressions on `astropy-12907`, but fails when an issue requires coordinated signature changes across multiple class methods (`RST.__init__` and `RST.write`).

---

### 6.3 Conclusion
This work presented a transparent, reproducible feasibility and component-ablation study of local post-training and agentic scaffolding for `google/gemma-2-2b-it` (2.61B) on a single 64 GB Apple M2 Max workstation. By separating syntactic validity (`21/25` = `84.0%` zero-shot SRI compliance on SWE-bench Lite), optimization telemetry (DR-LoRA `-7.48%` loss reduction; Step-Level Error Masking `45.7x` blunder probability suppression), multi-group advantage-estimator dynamics across $M=10$ bug tasks $\times 3$ seeds (demonstrating that naive length tie-breaking degrades executable unit-test resolution from `26.67%` to `0.00%`, whereas zero-leakage observable-context Rubric PRM substitution increases unit-test resolution to `66.67%`), and paired end-to-end sandbox verification ($N=4$, showing that 1-round DR-LoRA SFT [`C4a`] and 3-step GRPO-AVSPO [`C4b`] apply `3/4` patches cleanly and resolve `1/4` tasks, while 2-round test-guided failure distillation and PDR [`C5`] resolves `2/4` tasks [`50.0%`, 95% CI: `6.8%–93.2%`]), our study establishes both the local feasibility and the current empirical boundary conditions of sub-3B software engineering agents on consumer hardware.

---

## 7. References (Primary Peer-Reviewed Literature & Verified Technical Reports)

1. **Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K.** (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* International Conference on Learning Representations (ICLR 2024). [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
2. **Gemma Team, Google DeepMind.** (2024). *Gemma 2: Improving Open Language Models at a Practical Size.* Technical Report. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)
3. **Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O.** (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* Advances in Neural Information Processing Systems (NeurIPS 2024). [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
4. **Xia, C. S., Deng, Y., Dunn, S., & Zhang, L.** (2024). *Agentless: Demystifying LLM-based Software Engineering Agents.* [arXiv:2407.01489](https://arxiv.org/abs/2407.01489)
5. **Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., Pan, J., Song, Y., Li, B., Singh, J., et al.** (2024). *OpenHands: An Open Platform for AI Software Developers as Generalist Agents.* International Conference on Learning Representations (ICLR 2025). [arXiv:2407.16741](https://arxiv.org/abs/2407.16741)
6. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR 2022). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
7. **Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W., & Zhao, T.** (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* International Conference on Learning Representations (ICLR 2023). [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
8. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D.** (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
9. **DeepSeek-AI.** (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
10. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
11. **Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K.** (2023). *Let's Verify Step by Step.* International Conference on Learning Representations (ICLR 2024). [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
12. **Snell, C., Lee, J., Xu, K., & Kumar, A.** (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* International Conference on Learning Representations (ICLR 2025). [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
13. **Aleithan, R., Nadeem, M., & Hassan, A. E.** (2024). *SWE-Bench+: Enhanced Coding Benchmark for LLMs.* [OpenReview:R40rS2afQ3](https://openreview.net/forum?id=R40rS2afQ3)
14. **OpenAI.** (2024). *Introducing SWE-bench Verified.* OpenAI Preparedness Technical Report. [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/)
15. **Shinwari, H. U. K., & Usama, M.** (2025). *ARD-LoRA: Dynamic Rank Allocation for Parameter-Efficient Fine-Tuning of Foundation Models with Heterogeneous Adaptation Needs.* [arXiv:2506.18267](https://arxiv.org/abs/2506.18267)
16. **Tao, C., Chen, J., Jiang, Y., Kou, K., Wang, S., Wang, R., Li, X., Yang, S., Du, Y., Dai, J., Mao, Z., Wang, X., Shang, L., & Bai, H.** (2026). *SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving.* [arXiv:2601.01426](https://arxiv.org/abs/2601.01426)
17. **He, X., Sun, Q., Cheng, A., Li, X., Ji, X., Lu, H., Huang, R., & Hu, Q.** (2025/2026). *Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation.* International Conference on Machine Learning (ICML). [arXiv:2605.21125](https://arxiv.org/abs/2605.21125)