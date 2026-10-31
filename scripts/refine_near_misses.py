#!/usr/bin/env python3
"""Targeted 2nd-Round Refinement for Near-Miss Instances on SWE-bench Lite.

Focuses on instances that fixed the primary bug (passed FAIL_TO_PASS) or applied
cleanly but triggered a regression in PASS_TO_PASS (such as pytest-dev__pytest-11143).
Uses RepoGraph Doctor AST context and test-guided failure distillation to produce
a regression-free Round-2 fix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rich.console import Console
from rich.table import Table

from src.data.sri_formatter import SRIBlock, SRIFormatter
from src.evaluation.metrics import InstanceEvaluationResult
from src.evaluation.parallel_distill_refine import ParallelDistillRefine
from src.evaluation.patch_applicator import PatchApplicator
from src.evaluation.repo_manager import RepoManager
from src.evaluation.swe_bench_loader import SWEBenchInstance, SWEBenchLoader
from src.evaluation.swe_bench_runner import SWEBenchRunner
from src.peft.dr_lora import DRLoRAManager
from src.rlvr.verifier import DockerSandboxVerifier, ExecutionSandboxVerifier
from src.train import get_device, load_model_and_tokenizer, load_yaml

console = Console()


def refine_instance(
    instance: SWEBenchInstance,
    repo_dir: str,
    runner: SWEBenchRunner,
    generate_fn: Any,
    k: int = 4,
) -> InstanceEvaluationResult:
    """Run 2-Round Test-Guided Refinement on a specific instance."""
    console.print(f"\n[bold cyan]═══ Refining Instance: {instance.instance_id} ═══[/bold cyan]")
    f2p_cmd = runner.build_test_command(instance, is_pass_to_pass=False)
    p2p_cmd = runner.build_test_command(instance, is_pass_to_pass=True) if instance.pass_to_pass else None

    # Step 1: Initial Prompt with Code-Graph Context
    from src.agent.tools import RepoTools
    graph_ctx = RepoTools.get_graph_context(repo_dir, "rewrite", max_symbols=3)
    if "No code-graph" in graph_ctx:
        graph_ctx = RepoTools.get_graph_context(repo_dir, instance.instance_id.split("-")[-1], max_symbols=3)

    user_prompt = (
        f"{runner.SYSTEM_PROMPT}\n\n"
        f"Repository: {instance.repo}\n"
        f"Base Commit: {instance.base_commit}\n"
        f"Problem Statement:\n{instance.problem_statement}\n\n"
        f"Repository AST Code-Graph Context:\n{graph_ctx}\n\n"
        f"Output your fix in Search-and-Replace Infilling (SRI) format:"
    )

    console.print("[dim]Sampling Round-1 candidate fixes...[/dim]")
    round1_candidates = [generate_fn(user_prompt) for _ in range(k)]

    pdr = ParallelDistillRefine(k=k, max_refine_rounds=1)
    sb = ExecutionSandboxVerifier(timeout_seconds=120)

    # Evaluate Round 1 candidates
    scored_cands = []
    for c_idx, raw_out in enumerate(round1_candidates):
        blocks = SRIFormatter.parse(raw_out)
        res = sb.verify_patch(repo_dir, blocks, f2p_cmd)
        p2p_passed = True
        p2p_err = ""
        if res.passed and p2p_cmd:
            p2p_res = sb.verify_patch(repo_dir, blocks, p2p_cmd)
            p2p_passed = p2p_res.passed
            p2p_err = p2p_res.stdout or p2p_res.stderr

        cand = pdr.score_candidate(
            candidate_id=c_idx + 1,
            raw_output=raw_out,
            verify_fn=lambda b: (res.passed and p2p_passed, res.stdout, res.stderr or p2p_err),
        )
        scored_cands.append(cand)

    winning = max(scored_cands, key=lambda c: c.composite_score)
    if winning.passed_tests:
        console.print("[bold green]✓ Round 1 candidate resolved both FAIL_TO_PASS and PASS_TO_PASS![/bold green]")
        return runner.evaluate_instance(instance, repo_dir, generated_patch=winning.raw_output)

    # Step 2: Distill Failure Summary for Round 2
    distilled_feedback = pdr.distill_failure_analysis(scored_cands)
    console.print(f"[yellow]Round 1 did not fully pass. Distilled Feedback:\n{distilled_feedback}[/yellow]")

    round2_prompt = (
        f"{user_prompt}\n\n"
        f"{distilled_feedback}\n\n"
        f"Please refine your patch to resolve the bug without breaking existing tests in PASS_TO_PASS:"
    )

    console.print("[dim]Sampling Round-2 refined candidate fixes...[/dim]")
    round2_candidates = [generate_fn(round2_prompt) for _ in range(k)]
    scored_r2 = []
    for c_idx, raw_out in enumerate(round2_candidates):
        blocks = SRIFormatter.parse(raw_out)
        res = sb.verify_patch(repo_dir, blocks, f2p_cmd)
        p2p_passed = True
        p2p_err = ""
        if res.passed and p2p_cmd:
            p2p_res = sb.verify_patch(repo_dir, blocks, p2p_cmd)
            p2p_passed = p2p_res.passed
            p2p_err = p2p_res.stdout or p2p_res.stderr

        cand = pdr.score_candidate(
            candidate_id=c_idx + 1,
            raw_output=raw_out,
            verify_fn=lambda b: (res.passed and p2p_passed, res.stdout, res.stderr or p2p_err),
        )
        scored_r2.append(cand)

    winning_r2 = max(scored_r2, key=lambda c: c.composite_score)
    final_patch = winning_r2.raw_output if winning_r2.composite_score > winning.composite_score else winning.raw_output

    eval_res = runner.evaluate_instance(instance, repo_dir, generated_patch=final_patch)
    if eval_res.resolved:
        console.print(f"[bold green]✓ Successfully RESOLVED on Round 2![/bold green]")
    else:
        console.print(f"[bold red]✗ Round 2 completed: {eval_res.error_message}[/bold red]")
    return eval_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted 2nd-Round Refinement for Near-Miss Instances")
    parser.add_argument(
        "--instance",
        type=str,
        default="pytest-dev__pytest-11143",
        help="Target instance ID to refine (default: pytest-dev__pytest-11143)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./results/swe_bench_lite_tts_results.json",
        help="Path to results file to update",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sft_dr_lora/adapter_model.pt",
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of candidates per round")
    args = parser.parse_args()

    device = get_device()
    console.print(f"[bold]Target Instance:[/bold] {args.instance} on {device}")

    # Load instance
    all_instances = SWEBenchLoader.load_from_huggingface(limit=None)
    target = next((i for i in all_instances if i.instance_id == args.instance), None)
    if not target:
        console.print(f"[bold red]Instance '{args.instance}' not found in SWE-bench Lite![/bold red]")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it", device)
    if os.path.exists(args.checkpoint):
        DRLoRAManager.from_checkpoint(model=model, checkpoint_path=args.checkpoint, device=device)

    def generate_fn(prompt: str) -> str:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            chat_p = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            chat_p = prompt
        inputs = tokenizer(chat_p, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    runner = SWEBenchRunner(model_generate_fn=generate_fn)
    sandbox_dir = RepoManager.create_instance_sandbox(target.repo, target.base_commit, test_patch=target.test_patch)

    try:
        res = refine_instance(target, str(sandbox_dir), runner, generate_fn, k=args.k)

        # Update results file if resolved
        if os.path.exists(args.results_file):
            with open(args.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, r in enumerate(data.get("results", [])):
                if r.get("instance_id") == target.instance_id:
                    data["results"][idx] = {
                        "instance_id": target.instance_id,
                        "resolved": res.resolved,
                        "patch_applied": res.patch_applied,
                        "format_type": res.format_type,
                        "fail_to_pass_passed": res.fail_to_pass_passed,
                        "pass_to_pass_passed": res.pass_to_pass_passed,
                        "execution_time_seconds": round(res.execution_time_seconds, 2),
                        "error_message": res.error_message,
                    }
            data["resolved_instances"] = sum(1 for r in data["results"] if r.get("resolved"))
            data["pass_at_1"] = data["resolved_instances"] / len(data["results"])
            with open(args.results_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            console.print(f"[bold green]Updated {args.results_file} (Resolved: {data['resolved_instances']}/300)[/bold green]")
    finally:
        RepoManager.remove_sandbox(sandbox_dir, target.repo)


if __name__ == "__main__":
    main()
