#!/usr/bin/env python3
"""Batch Competition Task Simulation Runner (Option B).

Runs multi-turn autonomous evaluation across a diverse batch of official Kaggle Gemma Challenge tasks
(covering FastAPI, Rich, Requests, and HTTPX), keeping the model in memory to evaluate consecutive tasks
efficiently, and produces comprehensive aggregate statistics, pass rates, and diff summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.kaggle_submission_notebook import SandboxedHarness, find_input_paths, parse_tool_call
from src.peft.dr_lora import DRLoRAManager
from src.train import get_device, load_model_and_tokenizer

console = Console()


def run_batch_simulation(
    task_indices: List[int],
    max_turns: int = 6,
    checkpoint: str = "checkpoints/rlvr_competition/adapter_model.pt",
    output_report: str = "results/batch_simulation_report.json",
) -> Dict[str, Any]:
    tasks_path, snapshots_dir, graphs_dir, sub_root = find_input_paths()

    console.print(Panel(
        f"[bold cyan]Batch Competition Simulation (Option B)[/bold cyan]\n"
        f"Tasks File: {tasks_path}\n"
        f"Snapshots Dir: {snapshots_dir}\n"
        f"Checkpoint: {checkpoint}\n"
        f"Target Instances: {len(task_indices)} tasks: {task_indices}",
        title="Configuration",
    ))

    # 1. Load all tasks
    all_tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_tasks.append(json.loads(line))

    selected_tasks = []
    for idx in task_indices:
        if idx < len(all_tasks):
            t = all_tasks[idx]
            t["_orig_idx"] = idx
            selected_tasks.append(t)
        else:
            console.print(f"[yellow]Task index {idx} out of range (max {len(all_tasks) - 1})[/yellow]")

    if not selected_tasks:
        raise ValueError("No valid tasks selected for batch simulation.")

    # 2. Load model & DR-LoRA weights ONCE in memory
    device = get_device()
    console.print(f"[cyan]Loading base model google/gemma-2-2b-it on {device}...[/cyan]")
    model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it", device)

    if checkpoint and os.path.exists(checkpoint):
        console.print(f"[cyan]Loading fine-tuned DR-LoRA checkpoint from {checkpoint}...[/cyan]")
        manager = DRLoRAManager.from_checkpoint(model, checkpoint, device=device)
        console.print(f"[green]Loaded {len(manager.adapted_layers)} adapted LoRA layers.[/green]")

    model.eval()

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    # 3. Iterate through batch
    results: List[Dict[str, Any]] = []
    total_tool_calls = 0
    tool_counts: Dict[str, int] = {}
    repo_results: Dict[str, Dict[str, int]] = {}

    batch_start = time.time()

    for i, task in enumerate(selected_tasks):
        orig_idx = task.get("_orig_idx", i)
        iid = task["instance_id"]
        repo = task.get("repo", "unknown")
        commit = task.get("base_commit", "")
        problem = task.get("problem_statement", "").strip()

        console.print(f"\n[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold green]")
        console.print(f"[bold]Evaluating [{i + 1}/{len(selected_tasks)}] Task {orig_idx}: {iid}[/bold] ({repo})")
        console.print(f"[dim]Issue: {problem[:150]}...[/dim]")

        task_start = time.time()
        workspace = Path(f"/tmp/sim_batch_{iid}")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # Unpack snapshot
        snapshot_tar = snapshots_dir / f"{iid}.tgz"
        if snapshot_tar.exists():
            with tarfile.open(snapshot_tar, "r:gz") as tar:
                tar.extractall(path=workspace)
        else:
            subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
            (workspace / "README.md").write_text(f"# {repo}\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=str(workspace), capture_output=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=str(workspace), capture_output=True)

        # Pre-computed AST graph
        graph_path = graphs_dir / f"{repo.split('/')[-1]}_{commit}.json"
        graph_data = None
        if graph_path.exists():
            try:
                with open(graph_path, "r", encoding="utf-8") as gf:
                    graph_data = json.load(gf)
            except Exception:
                pass

        harness = SandboxedHarness(workspace_dir=workspace, graph_data=graph_data)

        # Dialog initialization
        system_prompt = (
            "You are an expert autonomous software engineer assigned to resolve an issue in a repository efficiently and decisively.\n"
            "Use the provided tools (search_similar_code, get_code_neighbors, read_file, edit_file, run_command, submit_patch) to localize, repair, verify, and submit your patch.\n\n"
            f"Repository: {repo}\n"
            f"Issue Description:\n{problem[:500]}\n"
        )
        dialogue: List[str] = [system_prompt]
        turns_used = 0

        for turn in range(1, max_turns + 1):
            turns_used = turn
            prompt_text = "\n".join(dialogue) + "\n<action>\n"
            response = generate_fn(prompt_text)
            clean_resp = response.split("</action>")[0].strip()

            tool_call = parse_tool_call(clean_resp)
            if tool_call:
                tname, targs = tool_call
                tool_counts[tname] = tool_counts.get(tname, 0) + 1
                console.print(f"  Turn {turn}: Tool Call -> [cyan]{tname}[/cyan]({list(targs.keys())})")

                if tname == "read_file":
                    obs = harness.read_file(targs.get("filepath") or targs.get("path", ""), int(targs.get("start_line", 1)), int(targs.get("end_line", 50)))
                elif tname == "edit_file":
                    target_fp = targs.get("filepath") or targs.get("path", "")
                    if target_fp.startswith("tests/") or "test_" in target_fp:
                        obs = json.dumps({"status": "error", "error": f"Editing '{target_fp}' is forbidden. Per competition rules, test files cannot be modified. All fixes must be made to library source files."})
                    else:
                        obs = harness.edit_file(target_fp, targs.get("old_string", ""), targs.get("new_string", ""))
                elif tname == "write_file":
                    obs = harness.write_file(targs.get("filepath") or targs.get("path", ""), targs.get("content", ""))
                elif tname == "run_command":
                    obs = harness.run_command(targs.get("command") or targs.get("cmd", ""))
                elif tname == "get_code_neighbors":
                    obs = harness.get_code_neighbors(targs.get("node") or targs.get("node_name", ""), targs.get("edge_type"))
                elif tname == "search_similar_code":
                    obs = harness.search_similar_code(targs.get("query", ""))
                elif tname == "get_code_subgraph":
                    nodes = [n.strip() for n in (targs.get("nodes", "")).split(",") if n.strip()]
                    obs = harness.get_code_subgraph(nodes)
                elif tname == "get_status":
                    obs = harness.get_status()
                elif tname == "submit_patch":
                    harness.submit_patch()
                    break
                else:
                    obs = json.dumps({"status": "error", "error": f"Unknown tool '{tname}'"})

                dialogue.append(f"<action>\n{clean_resp}\n</action>")
                dialogue.append(f"<observation>\n{obs[:500]}\n</observation>")
            else:
                if "submit_patch" in clean_resp:
                    harness.submit_patch()
                    break
                dialogue.append(f"<action>\n{clean_resp}\n</action>")

        if not harness.patch_submitted:
            harness.submit_patch()

        patch = harness.submitted_patch
        patch_len = len(patch.strip())
        task_time = time.time() - task_start

        # Cleanup workspace
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

        task_res = {
            "task_index": orig_idx,
            "instance_id": iid,
            "repo": repo,
            "turns_used": turns_used,
            "tool_calls_used": harness.tool_calls_used,
            "patch_size": patch_len,
            "has_patch": patch_len > 0,
            "time_seconds": round(task_time, 2),
        }
        results.append(task_res)
        total_tool_calls += harness.tool_calls_used

        # Update per-repo stats
        if repo not in repo_results:
            repo_results[repo] = {"total": 0, "patched": 0, "tool_calls": 0}
        repo_results[repo]["total"] += 1
        if patch_len > 0:
            repo_results[repo]["patched"] += 1
        repo_results[repo]["tool_calls"] += harness.tool_calls_used

        status_style = "green" if patch_len > 0 else "yellow"
        console.print(f"  Result: [{status_style}]Patch Size: {patch_len} bytes[/] | Tools: {harness.tool_calls_used} | Time: {task_time:.1f}s")

    # 4. Compile Aggregate Summary
    total_time = time.time() - batch_start
    total_tasks = len(results)
    total_patched = sum(1 for r in results if r["has_patch"])
    patch_rate = (total_patched / total_tasks) * 100 if total_tasks else 0.0

    summary_table = Table(title="Batch Simulation Results", show_header=True)
    summary_table.add_column("Task ID", style="cyan")
    summary_table.add_column("Repo", style="magenta")
    summary_table.add_column("Turns", justify="right")
    summary_table.add_column("Tools", justify="right")
    summary_table.add_column("Patch Size", justify="right")
    summary_table.add_column("Time", justify="right", style="green")

    for r in results:
        summary_table.add_row(
            r["instance_id"],
            r["repo"].split("/")[-1],
            str(r["turns_used"]),
            str(r["tool_calls_used"]),
            f"{r['patch_size']} B",
            f"{r['time_seconds']}s",
        )

    console.print("\n")
    console.print(summary_table)

    # Tool Distribution Table
    tool_table = Table(title="Tool Invocation Distribution", show_header=True)
    tool_table.add_column("Tool Name", style="cyan")
    tool_table.add_column("Call Count", justify="right", style="green")
    tool_table.add_column("Share (%)", justify="right")

    for tname, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        share = (cnt / total_tool_calls) * 100 if total_tool_calls else 0.0
        tool_table.add_row(tname, str(cnt), f"{share:.1f}%")

    console.print("\n")
    console.print(tool_table)

    # Save JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total_tasks,
        "total_patched": total_patched,
        "patch_rate_percent": round(patch_rate, 2),
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_task": round(total_tool_calls / total_tasks, 2) if total_tasks else 0,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_task": round(total_time / total_tasks, 2) if total_tasks else 0,
        "tool_distribution": tool_counts,
        "repo_breakdown": repo_results,
        "task_results": results,
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(Panel(
        f"[bold green]Batch Simulation Complete![/bold green]\n"
        f"Tasks Evaluated: {total_tasks}\n"
        f"Total Tool Calls: {total_tool_calls}\n"
        f"Avg Duration / Task: {total_time / total_tasks:.1f}s\n"
        f"Detailed Report Saved To: {out_p}",
        title="Summary",
    ))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Batch Task Simulation")
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 10, 11, 12, 18, 19, 20, 77],
        help="List of task indices in tasks.jsonl to evaluate (covers FastAPI, Requests, Rich, and HTTPX)",
    )
    parser.add_argument("--max-turns", type=int, default=6, help="Max turns per task")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rlvr_competition/adapter_model.pt")
    parser.add_argument("--output", type=str, default="results/batch_simulation_report.json")
    args = parser.parse_args()

    run_batch_simulation(
        task_indices=args.task_indices,
        max_turns=args.max_turns,
        checkpoint=args.checkpoint,
        output_report=args.output,
    )


if __name__ == "__main__":
    main()
