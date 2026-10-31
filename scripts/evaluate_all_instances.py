#!/usr/bin/env python3
"""Run SWE-bench Evaluation Across All 300 Lite (or 500 Verified) Instances.

Features:
- Evaluates on full benchmark datasets (princeton-nlp/SWE-bench_Lite or SWE-bench_Verified).
- Supports Single-Turn, Agentic Multi-Turn, and Parallel-Distill-Refine (PDR / TTS).
- Incremental JSON result saving after *every* instance (no loss on interruption).
- Automatic resume capability (--resume) to continue where a previous run left off.
- Generates comprehensive summary tables with per-repository breakdowns.

Usage:
  # Fast single-turn syntactic & patch evaluation across all 300 instances:
  PYTHONPATH=. uv run python scripts/evaluate_all_instances.py --mode single_turn

  # Multi-turn agentic evaluation with local worktrees on all instances:
  PYTHONPATH=. uv run python scripts/evaluate_all_instances.py --mode agentic --checkpoint checkpoints/sft_dr_lora/adapter_model.pt

  # Run on first 25 instances with resume enabled:
  PYTHONPATH=. uv run python scripts/evaluate_all_instances.py --limit 25 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rich.console import Console
from rich.table import Table

from src.data.sri_formatter import SRIFormatter
from src.evaluation.metrics import InstanceEvaluationResult, MetricsCalculator
from src.evaluation.repo_manager import RepoManager
from src.evaluation.swe_bench_loader import SWEBenchInstance, SWEBenchLoader
from src.evaluation.swe_bench_runner import SWEBenchRunner
from src.peft.dr_lora import DRLoRAManager
from src.rlvr.verifier import DockerSandboxVerifier
from src.train import get_device, load_model_and_tokenizer, load_yaml

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Evaluation Across All Instances"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/swe_bench_eval_all.yaml",
        help="Path to YAML evaluation configuration",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. princeton-nlp/SWE-bench_Lite, princeton-nlp/SWE-bench_Verified)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned DR-LoRA / RLVR adapter checkpoint",
    )
    parser.add_argument(
        "--mode",
        choices=["single_turn", "agentic", "tts"],
        default="single_turn",
        help="Evaluation mode: single_turn (~3.8s/instance), agentic (~1m/instance), or tts (~2m/instance)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of instances to evaluate (omit or 0 for all instances)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index in the dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/swe_bench_lite_all_results.json",
        help="Path to output JSON results file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already evaluated instances",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Enable Docker container sandbox verification if Docker is running",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Maximum exploration turns for agentic mode (default: 4)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Parallel rollout candidates for TTS / PDR mode (default: 4)",
    )
    return parser.parse_args()


def load_existing_results(output_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing instance results from JSON file to support resuming."""
    path = Path(output_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            results_list = data.get("results", [])
            return {r["instance_id"]: r for r in results_list if "instance_id" in r}
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse existing results ({e}); starting fresh.[/yellow]")
        return {}


def save_checkpoint_json(
    output_path: str,
    dataset_name: str,
    mode: str,
    model_name: str,
    checkpoint_path: Optional[str],
    results: List[InstanceEvaluationResult],
    total_in_run: int,
) -> None:
    """Incrementally save results to disk with atomic write."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    resolved_count = sum(1 for r in results if r.resolved)
    applied_count = sum(1 for r in results if r.patch_applied)
    sri_count = sum(1 for r in results if r.format_type == "sri")
    evaluated_count = len(results)

    pass_at_1 = resolved_count / evaluated_count if evaluated_count > 0 else 0.0
    apply_rate = applied_count / evaluated_count if evaluated_count > 0 else 0.0
    sri_rate = sri_count / evaluated_count if evaluated_count > 0 else 0.0

    payload = {
        "dataset": dataset_name,
        "mode": mode,
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_instances_target": total_in_run,
        "evaluated_instances": evaluated_count,
        "resolved_instances": resolved_count,
        "unresolved_instances": evaluated_count - resolved_count,
        "pass_at_1": pass_at_1,
        "patch_application_rate": apply_rate,
        "syntactic_sri_validity_rate": sri_rate,
        "results": [
            {
                "instance_id": r.instance_id,
                "resolved": r.resolved,
                "patch_applied": r.patch_applied,
                "format_type": r.format_type,
                "fail_to_pass_passed": r.fail_to_pass_passed,
                "pass_to_pass_passed": r.pass_to_pass_passed,
                "execution_time_seconds": round(r.execution_time_seconds, 2),
                "error_message": r.error_message,
            }
            for r in results
        ],
    }

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temp_path.replace(path)


def print_repository_breakdown(results: List[InstanceEvaluationResult]) -> None:
    """Print per-repository resolution and patch application stats."""
    repo_stats = defaultdict(lambda: {"total": 0, "resolved": 0, "applied": 0, "sri": 0})
    for r in results:
        repo = r.instance_id.split("__")[0] if "__" in r.instance_id else "other"
        repo_stats[repo]["total"] += 1
        if r.resolved:
            repo_stats[repo]["resolved"] += 1
        if r.patch_applied:
            repo_stats[repo]["applied"] += 1
        if r.format_type == "sri":
            repo_stats[repo]["sri"] += 1

    table = Table(title="Repository Breakdown", show_header=True)
    table.add_column("Repository", style="cyan")
    table.add_column("Instances", justify="right")
    table.add_column("Resolved", justify="right", style="green")
    table.add_column("Pass@1", justify="right", style="bold green")
    table.add_column("Applied", justify="right", style="yellow")
    table.add_column("Apply Rate", justify="right")
    table.add_column("SRI Valid", justify="right", style="magenta")

    for repo, s in sorted(repo_stats.items(), key=lambda x: -x[1]["total"]):
        p1 = f"{(s['resolved'] / s['total']):.1%}" if s["total"] > 0 else "0.0%"
        app = f"{(s['applied'] / s['total']):.1%}" if s["total"] > 0 else "0.0%"
        sri = f"{(s['sri'] / s['total']):.1%}" if s["total"] > 0 else "0.0%"
        table.add_row(
            repo,
            str(s["total"]),
            str(s["resolved"]),
            p1,
            str(s["applied"]),
            app,
            sri,
        )

    console.print(table)


def main() -> None:
    args = parse_args()

    # Load base configuration
    cfg = load_yaml(args.config) if os.path.exists(args.config) else {}
    device = get_device()

    dataset_name = args.dataset or cfg.get("benchmark", {}).get("dataset_name", "princeton-nlp/SWE-bench_Lite")
    split = args.split or cfg.get("benchmark", {}).get("split", "test")
    model_name = cfg.get("model", {}).get("name_or_path", "google/gemma-2-2b-it")
    adapter_checkpoint = args.checkpoint or cfg.get("model", {}).get("adapter_path")
    output_path = args.output or cfg.get("benchmark", {}).get("output_file", "./results/swe_bench_lite_all_results.json")

    limit_val = args.limit if (args.limit is not None and args.limit > 0) else None

    console.print(f"\n[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    console.print(f"[bold magenta]  SWE-bench Full Evaluation: {dataset_name} ({split})[/bold magenta]")
    console.print(f"[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]")
    console.print(f"[dim]• Mode: [bold]{args.mode.upper()}[/bold][/dim]")
    console.print(f"[dim]• Device: [bold]{device}[/bold][/dim]")
    console.print(f"[dim]• Model: [bold]{model_name}[/bold][/dim]")
    console.print(f"[dim]• Checkpoint: [bold]{adapter_checkpoint or 'None (Zero-Shot Base)'}[/bold][/dim]")
    console.print(f"[dim]• Output Destination: [bold]{output_path}[/bold][/dim]\n")

    # 1. Load instances
    console.print(f"[cyan]Fetching instances from Hugging Face dataset '{dataset_name}'...[/cyan]")
    all_instances = SWEBenchLoader.load_from_huggingface(dataset_name=dataset_name, split=split, limit=None)
    console.print(f"[green]Total available instances in split: {len(all_instances)}[/green]")

    # Slice instances according to start_idx and limit
    instances = all_instances[args.start_idx :]
    if limit_val is not None:
        instances = instances[:limit_val]

    total_instances = len(instances)
    console.print(f"[bold]Evaluating {total_instances} instances (indices {args.start_idx} to {args.start_idx + total_instances - 1})...[/bold]\n")

    # 2. Check for existing results to resume
    existing_map = {}
    if args.resume:
        existing_map = load_existing_results(output_path)
        console.print(f"[cyan]Loaded {len(existing_map)} previously evaluated instances from {output_path}.[/cyan]")

    # 3. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # 4. Load DR-LoRA / RLVR checkpoint if provided
    if adapter_checkpoint and os.path.exists(adapter_checkpoint):
        console.print(f"[cyan]Loading adapter checkpoint from {adapter_checkpoint}...[/cyan]")
        manager = DRLoRAManager.from_checkpoint(model=model, checkpoint_path=adapter_checkpoint, device=device)
        console.print(f"[green]Successfully loaded {len(manager.adapted_layers)} adapted submodules.[/green]")
    elif adapter_checkpoint:
        console.print(f"[yellow]Notice: Checkpoint '{adapter_checkpoint}' not found on disk; proceeding with base model weights.[/yellow]")

    # Generation helpers
    def format_chat_prompt(text: str) -> str:
        if "<start_of_turn>" in text or "<bos>" in text:
            return text
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
        max_tokens = cfg.get("model", {}).get("max_new_tokens", 1024)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def sample_gen_fn(prompt: str, count: int) -> List[str]:
        chat_prompt = format_chat_prompt(prompt)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        max_tokens = cfg.get("model", {}).get("max_new_tokens", 1024)
        results = []
        for _ in range(count):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            results.append(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
        return results

    verifier = DockerSandboxVerifier(
        enable_docker=args.docker,
        timeout_seconds=cfg.get("benchmark", {}).get("timeout_seconds", 120),
    )
    runner = SWEBenchRunner(model_generate_fn=generate_fn, verifier=verifier)

    # 5. Main Evaluation Loop
    results: List[InstanceEvaluationResult] = []
    overall_start_time = time.time()

    # Pre-populate already evaluated instances if resuming
    for inst in instances:
        if inst.instance_id in existing_map:
            cached = existing_map[inst.instance_id]
            results.append(
                InstanceEvaluationResult(
                    instance_id=cached["instance_id"],
                    resolved=cached.get("resolved", False),
                    patch_applied=cached.get("patch_applied", False),
                    format_type=cached.get("format_type", "text"),
                    fail_to_pass_passed=cached.get("fail_to_pass_passed", False),
                    pass_to_pass_passed=cached.get("pass_to_pass_passed", False),
                    execution_time_seconds=cached.get("execution_time_seconds", 0.0),
                    error_message=cached.get("error_message"),
                )
            )

    console.print(f"[bold green]Starting evaluation loop... ({len(results)}/{total_instances} already cached)[/bold green]\n")

    for idx, inst in enumerate(instances):
        # Skip if already in results
        if any(r.instance_id == inst.instance_id for r in results):
            continue

        item_idx = len(results) + 1
        elapsed_total = time.time() - overall_start_time
        avg_time = elapsed_total / item_idx if item_idx > 1 else 0.0
        remaining_sec = avg_time * (total_instances - item_idx)
        eta_min = remaining_sec / 60.0

        console.print(
            f"[bold cyan][{item_idx}/{total_instances}][/bold cyan] "
            f"[bold]{inst.instance_id}[/bold] "
            f"([dim]Repo: {inst.repo} | ETA: {eta_min:.1f}m[/dim])"
        )

        item_start = time.time()
        res: InstanceEvaluationResult

        try:
            if args.mode == "single_turn":
                prompt = runner.format_instance_prompt(inst)
                model_output = generate_fn(prompt)
                blocks = SRIFormatter.parse(model_output)
                has_sri = len(blocks) > 0
                format_type = "sri" if has_sri else ("diff" if "diff --git" in model_output else "text")

                # Test patch extraction validity
                res = InstanceEvaluationResult(
                    instance_id=inst.instance_id,
                    resolved=False,  # Single-turn without repository tools does not run sandboxed tests
                    patch_applied=has_sri,
                    format_type=format_type,
                    fail_to_pass_passed=False,
                    pass_to_pass_passed=False,
                    execution_time_seconds=time.time() - item_start,
                )

            elif args.mode == "agentic":
                sandbox_dir = None
                try:
                    sandbox_dir = RepoManager.create_instance_sandbox(
                        inst.repo, inst.base_commit, test_patch=inst.test_patch
                    )
                    repo_target = str(sandbox_dir)
                except Exception as e:
                    console.print(f"  [yellow]Sandbox checkout note: {e}[/yellow]")
                    repo_target = "."

                try:
                    res = runner.evaluate_instance_agentic(
                        instance=inst,
                        repo_dir=repo_target,
                        max_turns=args.max_turns,
                        tokenizer=tokenizer,
                    )
                finally:
                    if sandbox_dir:
                        RepoManager.remove_sandbox(sandbox_dir, inst.repo)

            elif args.mode == "tts":
                sandbox_dir = None
                try:
                    sandbox_dir = RepoManager.create_instance_sandbox(
                        inst.repo, inst.base_commit, test_patch=inst.test_patch
                    )
                    repo_target = str(sandbox_dir)
                except Exception:
                    repo_target = "."

                try:
                    res, _, _ = runner.evaluate_instance_pdr(
                        instance=inst,
                        repo_dir=repo_target,
                        k=args.k,
                        sample_generate_fn=sample_gen_fn,
                    )
                finally:
                    if sandbox_dir:
                        RepoManager.remove_sandbox(sandbox_dir, inst.repo)

        except Exception as e:
            console.print(f"  [bold red]Error evaluating instance {inst.instance_id}: {e}[/bold red]")
            res = InstanceEvaluationResult(
                instance_id=inst.instance_id,
                resolved=False,
                patch_applied=False,
                format_type="error",
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                execution_time_seconds=time.time() - item_start,
                error_message=str(e),
            )

        results.append(res)

        # Print item status
        if res.resolved:
            status = "[bold green]✓ RESOLVED[/bold green]"
        elif res.patch_applied:
            status = "[yellow]✓ PATCH APPLIED[/yellow]"
        elif res.format_type == "sri":
            status = "[cyan]✓ SRI DETECTED[/cyan]"
        else:
            status = "[red]✗ NO PATCH[/red]"

        console.print(f"  ↳ Status: {status} | Duration: {res.execution_time_seconds:.2f}s")

        # Save checkpoint JSON after *every* instance
        save_checkpoint_json(
            output_path=output_path,
            dataset_name=dataset_name,
            mode=args.mode,
            model_name=model_name,
            checkpoint_path=adapter_checkpoint,
            results=results,
            total_in_run=total_instances,
        )

    # 6. Final Summary and Leaderboard Reporting
    console.print(f"\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  Evaluation Completed Successfully![/bold green]")
    console.print(f"[bold green]═══════════════════════════════════════════════════════════[/bold green]\n")

    summary = MetricsCalculator.compute_summary(results)
    summary.print_summary()
    print_repository_breakdown(results)

    console.print(f"\n[bold cyan]Full report saved to:[/bold cyan] {output_path}")


if __name__ == "__main__":
    main()
