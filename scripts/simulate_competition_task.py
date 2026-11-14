#!/usr/bin/env python3
"""Local Competition Task Simulator.

Simulates the Kaggle Gemma Challenge competition evaluation harness on a local machine:
1. Loads an official task from /Users/vishnuanirudh/data/gemma-4-developer-agent/tasks.jsonl.
2. Unpacks the snapshot into an isolated sandbox worktree (or sets up temporary repo).
3. Binds the 9 official competition tools:
   - read_file, edit_file, write_file
   - run_command, get_status, submit_patch
   - get_code_neighbors, search_similar_code, get_code_subgraph
4. Runs the multi-turn agent with our fine-tuned DR-LoRA model.
5. Captures and verifies the final submitted patch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent_loop import MultiTurnAgent
from src.data.sri_formatter import SRIFormatter
from src.peft.dr_lora import DRLoRAManager
from src.train import get_device, load_model_and_tokenizer

console = Console()

DEFAULT_TASKS_FILE = "/Users/vishnuanirudh/data/gemma-4-developer-agent/tasks.jsonl"
DEFAULT_SNAPSHOTS_DIR = "/Users/vishnuanirudh/data/gemma-4-developer-agent/snapshots"
DEFAULT_GRAPHS_DIR = "/Users/vishnuanirudh/data/gemma-4-developer-agent/graphs"


class CompetitionHarnessContext:
    """Manages the 9 sandboxed competition tools and budget tracking."""

    def __init__(
        self,
        workspace_dir: Path,
        graph_data: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 25,
    ):
        self.workspace_dir = workspace_dir
        self.graph_data = graph_data or {"nodes": {}, "edges": []}
        self.max_tool_calls = max_tool_calls
        self.tool_calls_used = 0
        self.patch_submitted = False
        self.submitted_patch = ""

    def read_file(self, filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        self.tool_calls_used += 1
        target = self.workspace_dir / filepath.lstrip("/")
        if not target.exists() or not target.is_file():
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found"})

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            s_idx = max(0, (start_line or 1) - 1)
            e_idx = min(total, (end_line or (s_idx + 150)))
            content = "\n".join(lines[s_idx:e_idx])
            return json.dumps({
                "status": "ok",
                "filepath": filepath,
                "content": content,
                "start_line": s_idx + 1,
                "end_line": e_idx,
                "total_lines": total,
            })
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        self.tool_calls_used += 1
        target = self.workspace_dir / filepath.lstrip("/")
        if not target.exists():
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found"})

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            occurrences = content.count(old_string)
            if occurrences == 0:
                # Flexible line-by-line fallback
                old_lines = [l.strip() for l in old_string.splitlines() if l.strip()]
                target_lines = content.splitlines()
                matched_idx = -1
                for i in range(len(target_lines) - len(old_lines) + 1):
                    sub = [target_lines[i + j].strip() for j in range(len(old_lines))]
                    if sub == old_lines:
                        matched_idx = i
                        break
                if matched_idx >= 0:
                    before = target_lines[:matched_idx]
                    after = target_lines[matched_idx + len(old_lines):]
                    content = "\n".join(before + [new_string] + after)
                    target.write_text(content, encoding="utf-8")
                    return json.dumps({"status": "ok", "filepath": filepath, "strategy": "flexible", "occurrences": 1})
                return json.dumps({"status": "error", "error": "old_string pattern not found in file"})

            if occurrences > 1 and not allow_multiple:
                return json.dumps({"status": "error", "error": f"Multiple occurrences ({occurrences}) found without allow_multiple=True"})

            new_content = content.replace(old_string, new_string)
            target.write_text(new_content, encoding="utf-8")
            return json.dumps({"status": "ok", "filepath": filepath, "strategy": "exact", "occurrences": occurrences})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def write_file(self, filepath: str, content: str) -> str:
        self.tool_calls_used += 1
        target = self.workspace_dir / filepath.lstrip("/")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return json.dumps({"status": "ok", "filepath": filepath, "size": len(content)})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def run_command(self, command: str, timeout_seconds: int = 30) -> str:
        self.tool_calls_used += 1
        blocked = ["rm -rf /", "mkfs", ":(){ :|:& };:"]
        if any(b in command for b in blocked):
            return json.dumps({"status": "error", "error": "Blocked destructive command"})

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
            out = proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout
            err = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr
            return json.dumps({
                "status": "ok",
                "exit_code": proc.returncode,
                "stdout": out,
                "stderr": err,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": f"Command timed out after {timeout_seconds}s"})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_status(self) -> str:
        # Does not consume tool_calls budget
        return json.dumps({
            "status": "ok",
            "tool_calls_used": self.tool_calls_used,
            "max_tool_calls": self.max_tool_calls,
            "patch_submitted": self.patch_submitted,
            "patch_size": len(self.submitted_patch),
        })

    def submit_patch(self) -> str:
        # Does not consume tool_calls budget
        try:
            subprocess.run(["git", "add", "-N", "."], cwd=str(self.workspace_dir), capture_output=True)
            proc = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.submitted_patch = proc.stdout
            self.patch_submitted = True
            return json.dumps({
                "status": "ok",
                "patch_size": len(self.submitted_patch),
                "is_empty": len(self.submitted_patch.strip()) == 0,
            })
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_code_neighbors(self, node: str, edge_type: Optional[str] = None, max_neighbors: int = 20) -> str:
        self.tool_calls_used += 1
        nodes = self.graph_data.get("nodes", {})
        edges = self.graph_data.get("edges", [])

        # Match symbol
        matched_id = None
        for nid in nodes:
            if node == nid or node.lower() in nid.lower():
                matched_id = nid
                break

        if not matched_id:
            return json.dumps({"status": "ok", "node": node, "neighbors": [], "count": 0})

        neighbors = []
        for e in edges:
            if e.get("from") == matched_id or e.get("to") == matched_id:
                other = e.get("to") if e.get("from") == matched_id else e.get("from")
                if edge_type and e.get("type", "").lower() != edge_type.lower():
                    continue
                neighbors.append(other)

        neighbors = list(dict.fromkeys(neighbors))[:max_neighbors]
        return json.dumps({"status": "ok", "node": matched_id, "neighbors": neighbors, "count": len(neighbors)})

    def search_similar_code(self, query: str, k: int = 5) -> str:
        self.tool_calls_used += 1
        nodes = self.graph_data.get("nodes", {})
        results = []
        clean_q = query.lower().strip()

        for nid, ndata in nodes.items():
            name = ndata.get("name", "")
            code_snip = ndata.get("code", "")[:200]
            score = 0.0
            if clean_q == name.lower():
                score = 1.0
            elif clean_q in name.lower():
                score = 0.8
            elif clean_q in code_snip.lower():
                score = 0.5

            if score > 0:
                results.append({"node_name": nid, "code": code_snip, "similarity": score})

        results.sort(key=lambda x: -x["similarity"])
        return json.dumps({"status": "ok", "query": query, "results": results[:k], "count": len(results[:k])})

    def get_code_subgraph(self, nodes: List[str]) -> str:
        self.tool_calls_used += 1
        all_edges = self.graph_data.get("edges", [])
        node_set = set(nodes)
        sub_edges = [
            e for e in all_edges if e.get("from") in node_set and e.get("to") in node_set
        ]
        return json.dumps({
            "status": "ok",
            "nodes": list(node_set),
            "edges": sub_edges,
            "node_count": len(node_set),
            "edge_count": len(sub_edges),
        })


def run_simulation(
    task_idx: int = 0,
    instance_id: Optional[str] = None,
    checkpoint: str = "checkpoints/rlvr_competition/adapter_model.pt",
    tasks_file: str = DEFAULT_TASKS_FILE,
    snapshots_dir: str = DEFAULT_SNAPSHOTS_DIR,
    graphs_dir: str = DEFAULT_GRAPHS_DIR,
    max_turns: int = 4,
) -> Dict[str, Any]:
    """Execute end-to-end task simulation against the local sandbox."""
    tasks_path = Path(tasks_file)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")

    # 1. Load target task
    selected_task = None
    with open(tasks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            if instance_id and data.get("instance_id") == instance_id:
                selected_task = data
                task_idx = idx
                break
            elif not instance_id and idx == task_idx:
                selected_task = data
                break

    if not selected_task:
        raise ValueError(f"Task index {task_idx} or instance_id '{instance_id}' not found.")

    iid = selected_task.get("instance_id")
    repo = selected_task.get("repo")
    commit = selected_task.get("base_commit", "")
    problem = selected_task.get("problem_statement", "")

    console.print(Panel(
        f"[bold cyan]Task {task_idx}: {iid}[/bold cyan] ({repo})\n"
        f"Base Commit: {commit[:10]}\n"
        f"Problem: {problem[:200]}...",
        title="Simulated Competition Task",
    ))

    # 2. Setup workspace directory
    workspace = Path(f"./tmp/sim_{iid}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Check for snapshot tarball
    snapshot_tar = Path(snapshots_dir) / f"{iid}.tgz"
    if snapshot_tar.exists():
        console.print(f"[cyan]Unpacking snapshot {snapshot_tar.name}...[/cyan]")
        with tarfile.open(snapshot_tar, "r:gz") as tar:
            tar.extractall(path=workspace)
        console.print("[green]Snapshot unpacked successfully.[/green]")
    else:
        # Fallback: initialize empty git repo
        subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
        (workspace / "README.md").write_text(f"# Simulated {repo}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(workspace), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(workspace), capture_output=True)

    # 3. Load pre-computed graph data
    graph_path = Path(graphs_dir) / f"{repo.split('/')[-1]}_{commit}.json"
    graph_data = None
    if graph_path.exists():
        console.print(f"[cyan]Loading pre-computed AST graph from {graph_path.name}...[/cyan]")
        try:
            with open(graph_path, "r", encoding="utf-8") as gf:
                graph_data = json.load(gf)
            console.print(f"[green]Loaded graph with {len(graph_data.get('nodes', {}))} nodes.[/green]")
        except Exception as e:
            console.print(f"[yellow]Note loading graph: {e}[/yellow]")

    # 4. Initialize Harness Context
    harness = CompetitionHarnessContext(workspace_dir=workspace, graph_data=graph_data)

    # 5. Load model and DR-LoRA checkpoint
    device = get_device()
    console.print(f"[cyan]Loading base model on {device}...[/cyan]")
    model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it", device)

    if checkpoint and os.path.exists(checkpoint):
        console.print(f"[cyan]Loading fine-tuned DR-LoRA checkpoint from {checkpoint}...[/cyan]")
        manager = DRLoRAManager.from_checkpoint(model, checkpoint, device=device)
        console.print(f"[green]Loaded {len(manager.adapted_layers)} LoRA layers.[/green]")

    model.eval()

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    # 6. Execute multi-turn simulation
    console.print(f"\n[bold green]Starting Agent Exploration (Max Turns: {max_turns})...[/bold green]")
    history = [
        f"You are an autonomous SWE agent fixing a bug in {repo}.\n"
        f"Issue Description:\n{problem[:600]}\n"
        "Explore the repository with read_file, search_similar_code, or run_command, "
        "and when ready, output your fix using Search-and-Replace Infilling (SRI) format or submit_patch()."
    ]

    for turn in range(1, max_turns + 1):
        console.print(f"\n[bold]Turn {turn}/{max_turns}:[/bold] Generating agent response...")
        prompt_text = "\n\n".join(history)
        response = generate_fn(prompt_text)
        console.print(f"[dim]Agent Response:[/dim]\n{response[:300]}...")

        # Check for tool call
        tool_call = MultiTurnAgent.parse_tool_call(response)
        if tool_call:
            tname, targs = tool_call
            console.print(f"[bold cyan]Tool Call Detected:[/bold cyan] {tname}({targs})")
            if tname == "read_file":
                obs = harness.read_file(targs.get("filepath") or targs.get("path", ""), int(targs.get("start_line", 1)), int(targs.get("end_line", 50)))
            elif tname == "edit_file":
                obs = harness.edit_file(targs.get("filepath") or targs.get("path", ""), targs.get("old_string", ""), targs.get("new_string", ""))
            elif tname == "write_file":
                obs = harness.write_file(targs.get("filepath") or targs.get("path", ""), targs.get("content", ""))
            elif tname == "run_command":
                obs = harness.run_command(targs.get("command") or targs.get("cmd", ""))
            elif tname == "get_code_neighbors":
                obs = harness.get_code_neighbors(targs.get("node", ""), targs.get("edge_type"))
            elif tname == "search_similar_code":
                obs = harness.search_similar_code(targs.get("query", ""))
            elif tname == "submit_patch":
                obs = harness.submit_patch()
            else:
                obs = json.dumps({"status": "error", "error": f"Unknown tool '{tname}'"})

            history.append(f"Action: <tool_call name=\"{tname}\" />")
            history.append(f"Observation: {obs[:500]}")
        else:
            # Check for SRI block or patch submission
            sri_blocks = SRIFormatter.parse(response)
            if sri_blocks:
                console.print(f"[bold green]Detected {len(sri_blocks)} SRI replacement block(s)![/bold green]")
                for b in sri_blocks:
                    harness.edit_file(b.file_path, b.search_content, b.replace_content)
                harness.submit_patch()
                break

            if "submit_patch" in response:
                harness.submit_patch()
                break

            history.append(response)

    # 7. Collect results
    if not harness.patch_submitted:
        harness.submit_patch()

    patch = harness.submitted_patch
    patch_size = len(patch)
    console.print(Panel(
        f"Patch Submitted: {harness.patch_submitted}\n"
        f"Total Tool Calls: {harness.tool_calls_used}\n"
        f"Generated Patch Size: {patch_size} bytes\n"
        f"Patch Preview:\n{patch[:400] if patch else '(No git diff produced)'}",
        title="Simulation Results",
    ))

    # Clean up workspace
    if workspace.exists():
        shutil.rmtree(workspace)

    return {
        "instance_id": iid,
        "repo": repo,
        "tool_calls_used": harness.tool_calls_used,
        "patch_submitted": harness.patch_submitted,
        "patch_size": patch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Competition Task Execution")
    parser.add_argument("--task-index", type=int, default=0, help="Task index in tasks.jsonl")
    parser.add_argument("--instance-id", type=str, default=None, help="Target specific instance_id")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rlvr_competition/adapter_model.pt")
    parser.add_argument("--max-turns", type=int, default=3, help="Max agent turns")
    args = parser.parse_args()

    run_simulation(
        task_idx=args.task_index,
        instance_id=args.instance_id,
        checkpoint=args.checkpoint,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
