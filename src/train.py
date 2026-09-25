"""Unified CLI Entrypoint for Gemma Autonomous SWE: SFT (DR-LoRA), RLVR (GRPO+AVSPO), and SWE-bench Evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import torch
import yaml
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env
load_dotenv()

from src.data.curriculum import CurriculumTrajectoryDataset
from src.data.masking import StepLevelErrorMasker, TrajectoryStep
from src.data.sri_formatter import SRIFormatter
from src.evaluation.metrics import (
    BenchmarkSummary,
    InstanceEvaluationResult,
    MetricsCalculator,
)
from src.evaluation.patch_applicator import PatchApplicator
from src.evaluation.swe_bench_loader import SWEBenchInstance, SWEBenchLoader
from src.evaluation.swe_bench_runner import SWEBenchRunner
from src.peft.dr_lora import DRLoRAManager
from src.rlvr.avspo import AVSPOAdvantageEstimator
from src.rlvr.grpo import GRPOTrainer, GRPOTrainingConfig
from src.rlvr.prm import RubricProcessRewardModel

console = Console()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Detect optimal Apple Silicon or CUDA device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load base Gemma causal LM and tokenizer."""
    console.print(f"[cyan]Loading model '{model_name}' on {device}...[/cyan]")
    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 or float16 for Apple Silicon MPS
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    ).to(device)

    console.print(f"[green]Model loaded successfully on {device} ({dtype}).[/green]")
    return model, tokenizer


def run_sft(
    config_path: str,
    epochs: int = 1,
    save_checkpoint: str = "./checkpoints/sft_dr_lora/adapter_model.pt",
) -> None:
    """Execute multi-tier curriculum SFT with Dynamic Rank LoRA and Step-Level Error Masking."""
    cfg = load_yaml(config_path)
    device = get_device()
    model_name = cfg["model"]["name_or_path"]

    console.print(f"[bold green]Starting SFT with DR-LoRA[/bold green] on {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # 1. Inject Dynamic Rank LoRA
    target_modules = cfg["peft"]["target_modules"]
    initial_rank = cfg["peft"]["initial_rank"]
    max_rank = cfg["peft"]["max_rank"]
    min_rank = cfg["peft"]["min_rank"]

    console.print(f"Injecting DR-LoRA adapters (Initial Rank: {initial_rank}, Max Rank: {max_rank})...")
    manager = DRLoRAManager(
        model=model,
        target_modules=target_modules,
        max_rank=max_rank,
        min_rank=min_rank,
        initial_rank=initial_rank,
        lora_alpha=cfg["peft"].get("lora_alpha", 16.0),
    )
    console.print(f"Adapted {len(manager.adapted_layers)} submodules.")

    # 2. Setup Step-Level Error Masker & Optimizer
    masker = StepLevelErrorMasker(tokenizer)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
    )

    # 3. Setup Multi-Tier Curriculum Dataset (Easy -> Medium -> Hard)
    curriculum = CurriculumTrajectoryDataset.create_synthetic_curriculum(masker)
    console.print(
        f"[cyan]Loaded curriculum dataset with {len(curriculum)} multi-tier trajectories (Easy -> Medium -> Hard).[/cyan]"
    )

    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for idx, batch in enumerate(curriculum):
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Track module saliency from parameter gradients
            manager.saliency_tracker.update_from_model(model)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # Dynamic rank reallocation with TV/EMA smoothing after curriculum pass
        new_ranks = manager.reallocate_ranks()
        avg_loss = epoch_loss / len(curriculum)
        console.print(
            f"[bold green]Curriculum Epoch {ep}/{epochs} Completed[/bold green] | Avg Masked Loss: {avg_loss:.4f}"
        )
        console.print("Sample Active Rank Allocations after Saliency Reallocation:")
        for mod_name, r in list(new_ranks.items())[:4]:
            console.print(f"  • {mod_name}: rank {r}")

    # 4. Save fine-tuned checkpoint and metadata
    manager.save_adapters(save_checkpoint)
    console.print(f"[bold green]Saved fine-tuned DR-LoRA checkpoint to {save_checkpoint}[/bold green]")


def run_rlvr(config_path: str) -> None:
    """Execute RLVR with Critic-Free GRPO + AVSPO Anti-Advantage Collapse."""
    cfg = load_yaml(config_path)
    device = get_device()
    model_name = cfg["model"]["name_or_path"]

    console.print(f"[bold cyan]Starting RLVR (GRPO + AVSPO)[/bold cyan] on {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    grpo_cfg = GRPOTrainingConfig(
        group_size=cfg["grpo"]["group_size"],
        epsilon_clip=cfg["grpo"]["epsilon_clip"],
        kl_beta=cfg["grpo"]["kl_beta"],
        learning_rate=float(cfg["grpo"]["learning_rate"]),
        max_new_tokens=cfg["grpo"].get("max_new_tokens", 256),
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_cfg,
        device=device,
    )
    prm = RubricProcessRewardModel()

    console.print(f"Sampling group rollouts (G={grpo_cfg.group_size}) for sample coding task...")
    prompt = (
        "You are an expert autonomous software engineer.\n"
        "Bug description: Function divide(a, b) does not handle b=0.\n"
        "Output your fix in Search-and-Replace Infilling (SRI) format:"
    )

    rollouts = trainer.generate_group_rollouts(prompt)
    console.print(f"[green]Generated {len(rollouts)} rollouts successfully.[/green]")

    # Evaluate rollouts against Rubric PRM and verifiable test
    process_scores = [prm.evaluate_trajectory(r).total_score for r in rollouts]

    # Demonstrate AVSPO: Simulate an all-failing or homogeneous batch
    simulated_test_rewards = [0.0] * len(rollouts)

    console.print("[cyan]Performing Critic-Free GRPO step with AVSPO Advantage Normalization...[/cyan]")
    metrics = trainer.train_step(
        prompt_text=prompt,
        rollouts=rollouts,
        execution_rewards=simulated_test_rewards,
        process_scores=process_scores,
    )

    console.print(f"[bold green]RLVR Step Completed![/bold green]")
    console.print(f"  • Mean Policy Loss: {metrics['mean_loss']:.4f}")
    console.print(f"  • Mean KL Penalty: {metrics['mean_kl']:.4f}")
    console.print(f"  • Advantage Collapse Neutralized: {bool(metrics['was_collapsed'])}")
    console.print(f"  • Advantage Collapse Rate (ACR): {metrics['advantage_collapse_rate']:.2%}")


def run_eval(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    agentic: bool = False,
    tts: bool = False,
    k: int = 4,
) -> None:
    """Execute SWE-bench Evaluation on Lite/Verified instances."""
    cfg = load_yaml(config_path)
    device = get_device()
    model_name = cfg["model"]["name_or_path"]
    dataset_name = cfg["benchmark"]["dataset_name"]
    limit = cfg["benchmark"].get("limit", 5)

    mode_label = "Agentic Multi-Turn" if agentic else ("Parallel-Distill-Refine (TTS)" if tts else "Single-Turn")
    console.print(
        f"[bold magenta]Starting SWE-bench Evaluation ({mode_label})[/bold magenta] on {dataset_name} (limit={limit})"
    )
    instances = SWEBenchLoader.load_from_huggingface(dataset_name=dataset_name, limit=limit)
    console.print(f"Loaded {len(instances)} instances from {dataset_name}.")

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Load DR-LoRA checkpoint if specified
    adapter_checkpoint = checkpoint_path or cfg["model"].get("adapter_path")
    if adapter_checkpoint and os.path.exists(adapter_checkpoint):
        console.print(f"[cyan]Loading fine-tuned DR-LoRA checkpoint from {adapter_checkpoint}...[/cyan]")
        manager = DRLoRAManager.from_checkpoint(model=model, checkpoint_path=adapter_checkpoint, device=device)
        console.print(f"[green]Successfully loaded {len(manager.adapted_layers)} adapted submodules.[/green]")

    def format_chat_prompt(text: str) -> str:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return text

    def generate_fn(prompt: str) -> str:
        chat_prompt = format_chat_prompt(prompt)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg["model"].get("max_new_tokens", 512),
                temperature=cfg["model"].get("temperature", 0.0),
                do_sample=cfg["model"].get("temperature", 0.0) > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def sample_gen_fn(prompt: str, count: int) -> List[str]:
        chat_prompt = format_chat_prompt(prompt)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        results = []
        for _ in range(count):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg["model"].get("max_new_tokens", 512),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            results.append(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
        return results

    runner = SWEBenchRunner(model_generate_fn=generate_fn)
    results: List[InstanceEvaluationResult] = []

    table = Table(title=f"SWE-bench Evaluation Results ({mode_label})", show_header=True)
    table.add_column("Instance ID", style="cyan", no_wrap=True)
    table.add_column("Repo", style="blue")
    table.add_column("Format Detected", style="yellow")
    table.add_column("SRI Blocks", style="magenta")
    table.add_column("Status", style="green")

    for idx, inst in enumerate(instances):
        console.print(f"[dim]Evaluating instance {idx+1}/{len(instances)}: {inst.instance_id}...[/dim]")
        start = time.time()

        if tts:
            # Parallel-Distill-Refine Test-Time Scaling
            from src.evaluation.repo_manager import RepoManager
            sandbox_dir = None
            try:
                sandbox_dir = RepoManager.create_instance_sandbox(inst.repo, inst.base_commit)
                repo_target = str(sandbox_dir)
            except Exception:
                repo_target = "."

            try:
                res, pass_at_k, feedback = runner.evaluate_instance_pdr(
                    instance=inst,
                    repo_dir=repo_target,
                    k=k,
                    sample_generate_fn=sample_gen_fn,
                )
            finally:
                if sandbox_dir:
                    RepoManager.remove_sandbox(sandbox_dir, inst.repo)

            has_sri = res.patch_applied
            format_type = res.format_type
            blocks = [1] if has_sri else []
        elif agentic:
            # Interactive multi-turn agent exploration with RepoTools
            from src.evaluation.repo_manager import RepoManager

            sandbox_dir = None
            try:
                console.print(f"[dim]Preparing sandbox for {inst.repo} @ {inst.base_commit[:8]}...[/dim]")
                sandbox_dir = RepoManager.create_instance_sandbox(inst.repo, inst.base_commit)
                repo_target = str(sandbox_dir)
            except Exception as e:
                console.print(f"[yellow]Repo checkout fallback ({e}); using local directory.[/yellow]")
                repo_target = "."

            try:
                res = runner.evaluate_instance_agentic(
                    instance=inst,
                    repo_dir=repo_target,
                    max_turns=cfg.get("agent", {}).get("max_turns", 4),
                    tokenizer=tokenizer,
                )
            finally:
                if sandbox_dir:
                    RepoManager.remove_sandbox(sandbox_dir, inst.repo)

            has_sri = res.patch_applied
            format_type = res.format_type
            blocks = [1] if has_sri else []
        else:
            prompt = runner.format_instance_prompt(inst)
            model_output = generate_fn(prompt)
            blocks = SRIFormatter.parse(model_output)
            has_sri = len(blocks) > 0
            format_type = "sri" if has_sri else ("diff" if "diff --git" in model_output else "text")

            res = InstanceEvaluationResult(
                instance_id=inst.instance_id,
                resolved=False,
                patch_applied=has_sri,
                format_type=format_type,
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                execution_time_seconds=time.time() - start,
            )

        results.append(res)
        table.add_row(
            inst.instance_id,
            inst.repo,
            format_type,
            str(len(blocks)),
            "[green]FORMATTED[/green]" if has_sri else "[red]NO SRI[/red]",
        )

    console.print(table)
    summary = MetricsCalculator.compute_summary(results)
    summary.print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Autonomous SWE Pipeline")
    parser.add_argument(
        "--mode",
        choices=["sft", "rlvr", "eval"],
        required=True,
        help="Pipeline mode to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of curriculum training epochs for SFT",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to save or load DR-LoRA checkpoint",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Enable multi-turn interactive agent evaluation",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable Parallel-Distill-Refine (PDR) Test-Time Scaling",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of parallel rollouts for TTS (default: 4)",
    )

    args = parser.parse_args()

    if args.mode == "sft":
        save_path = args.checkpoint or "./checkpoints/sft_dr_lora/adapter_model.pt"
        run_sft(args.config, epochs=args.epochs, save_checkpoint=save_path)
    elif args.mode == "rlvr":
        run_rlvr(args.config)
    elif args.mode == "eval":
        run_eval(
            args.config,
            checkpoint_path=args.checkpoint,
            agentic=args.agentic,
            tts=args.tts,
            k=args.k,
        )


if __name__ == "__main__":
    main()
