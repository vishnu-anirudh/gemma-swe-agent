#!/usr/bin/env python3
"""Interactive Autonomous Coding Agent CLI.

Use your post-trained Gemma SWE agent as a real-world coding assistant on any local repository.

Examples:
  # One-shot bug fix on current repo:
  uv run python scripts/run_agent.py --task "Fix bug in auth.py where user profile is None"

  # Run on another project with interactive turn approvals:
  uv run python scripts/run_agent.py --repo ~/code/my-app --interactive

  # Dry-run mode (previews proposed diff without modifying files):
  uv run python scripts/run_agent.py --task "Add docstrings to all functions in src/utils.py" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent_loop import MultiTurnAgent
from src.data.sri_formatter import SRIFormatter
from src.peft.dr_lora import DRLoRAManager
from src.train import get_device, load_model_and_tokenizer

console = Console()


class LocalRepoHarness:
    """Provides safe sandboxed file operations and command execution on a local repository."""

    def __init__(self, repo_dir: Path, dry_run: bool = False):
        self.repo_dir = repo_dir.resolve()
        self.dry_run = dry_run
        self.tool_calls_used = 0
        self.backup_files: Dict[Path, str] = {}
        self.modified_files: List[Path] = []

    def read_file(self, filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        self.tool_calls_used += 1
        target = self._resolve_path(filepath)
        if not target or not target.is_file():
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found in {self.repo_dir}"})

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            s_idx = max(0, (start_line or 1) - 1)
            e_idx = min(total, (end_line or (s_idx + 120)))
            content = "\n".join(lines[s_idx:e_idx])
            return json.dumps({
                "status": "ok",
                "filepath": str(target.relative_to(self.repo_dir)),
                "content": content,
                "start_line": s_idx + 1,
                "end_line": e_idx,
                "total_lines": total,
            })
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def edit_file(self, filepath: str, old_string: str, new_string: str) -> str:
        self.tool_calls_used += 1
        target = self._resolve_path(filepath)
        if not target or not target.is_file():
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found"})

        try:
            original = target.read_text(encoding="utf-8", errors="replace")
            if target not in self.backup_files:
                self.backup_files[target] = original

            if old_string not in original:
                # Flexible fallback: strip trailing whitespace per line
                old_lines = [l.strip() for l in old_string.splitlines() if l.strip()]
                target_lines = original.splitlines()
                match_idx = -1
                for i in range(len(target_lines) - len(old_lines) + 1):
                    window = [l.strip() for l in target_lines[i : i + len(old_lines)]]
                    if window == old_lines:
                        match_idx = i
                        break
                if match_idx >= 0:
                    before = "\n".join(target_lines[:match_idx])
                    after = "\n".join(target_lines[match_idx + len(old_lines) :])
                    new_content = (before + "\n" + new_string + "\n" + after).strip() + "\n"
                    if not self.dry_run:
                        target.write_text(new_content, encoding="utf-8")
                    if target not in self.modified_files:
                        self.modified_files.append(target)
                    return json.dumps({"status": "ok", "filepath": str(target.relative_to(self.repo_dir)), "strategy": "flexible_line_match"})
                return json.dumps({"status": "error", "error": f"Pattern to replace not found in {filepath}"})

            new_content = original.replace(old_string, new_string, 1)
            if not self.dry_run:
                target.write_text(new_content, encoding="utf-8")
            if target not in self.modified_files:
                self.modified_files.append(target)
            return json.dumps({"status": "ok", "filepath": str(target.relative_to(self.repo_dir)), "strategy": "exact"})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def run_command(self, command: str) -> str:
        self.tool_calls_used += 1
        blocked = ["rm -rf /", "mkfs", ":(){ :|:& };:"]
        if any(b in command for b in blocked):
            return json.dumps({"status": "error", "error": "Blocked destructive shell command"})

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=45,
            )
            return json.dumps({
                "status": "ok",
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout,
                "stderr": proc.stderr[-1000:] if len(proc.stderr) > 1000 else proc.stderr,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": "Command timed out after 45s"})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def search_similar_code(self, query: str, k: int = 5) -> str:
        self.tool_calls_used += 1
        clean_q = query.strip().lower()
        results = []

        # Fast search using ripgrep or python walk
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__pycache__", "node_modules", "venv", ".venv"))]
            for file in files:
                if file.endswith((".py", ".js", ".ts", ".rs", ".go", ".cpp", ".c", ".md", ".json")):
                    fp = Path(root) / file
                    rel_p = fp.relative_to(self.repo_dir)
                    try:
                        text = fp.read_text(encoding="utf-8", errors="ignore")
                        if clean_q in str(rel_p).lower():
                            results.append({"filepath": str(rel_p), "match_type": "filename", "snippet": text[:150]})
                        elif clean_q in text.lower():
                            # Extract matching line
                            for line in text.splitlines():
                                if clean_q in line.lower():
                                    results.append({"filepath": str(rel_p), "match_type": "content", "snippet": line.strip()[:150]})
                                    break
                    except Exception:
                        pass
                if len(results) >= k:
                    break
            if len(results) >= k:
                break

        return json.dumps({"status": "ok", "query": query, "results": results[:k], "count": len(results[:k])})

    def get_diff(self) -> str:
        """Return git diff of changes made."""
        try:
            subprocess.run(["git", "add", "-N", "."], cwd=str(self.repo_dir), capture_output=True)
            proc = subprocess.run(["git", "diff", "--color=never"], cwd=str(self.repo_dir), capture_output=True, text=True)
            return proc.stdout
        except Exception:
            return ""

    def rollback(self) -> None:
        """Rollback all modified files to original state."""
        for target, orig in self.backup_files.items():
            try:
                target.write_text(orig, encoding="utf-8")
            except Exception:
                pass

    def _resolve_path(self, filepath: str) -> Optional[Path]:
        clean = filepath.strip().lstrip("/")
        direct = self.repo_dir / clean
        if direct.exists():
            return direct
        # Candidate: dot-path to slash
        cand = self.repo_dir / (clean.replace(".", "/") + ".py")
        if cand.exists():
            return cand
        return None


def run_agent_session(
    repo_path: Path,
    task_description: str,
    checkpoint: str = "checkpoints/rlvr_competition/adapter_model.pt",
    max_turns: int = 8,
    interactive: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute autonomous agent loop on local repository."""
    console.print(Panel(
        f"[bold cyan]Local Autonomous Coding Agent[/bold cyan]\n"
        f"Target Repository: {repo_path.resolve()}\n"
        f"Task Description: {task_description}\n"
        f"Interactive Approvals: {interactive} | Dry-Run Mode: {dry_run}",
        title="Session Initialized",
    ))

    # Initialize model
    device = get_device()
    console.print(f"[cyan]Loading base model google/gemma-2-2b-it on {device}...[/cyan]")
    model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it", device)

    if checkpoint and os.path.exists(checkpoint):
        console.print(f"[cyan]Loading fine-tuned DR-LoRA weights from {checkpoint}...[/cyan]")
        manager = DRLoRAManager.from_checkpoint(model, checkpoint, device=device)
        console.print(f"[green]Attached {len(manager.adapted_layers)} LoRA layers.[/green]")
    model.eval()

    harness = LocalRepoHarness(repo_dir=repo_path, dry_run=dry_run)

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    system_prompt = (
        "You are an expert autonomous software engineer assigned to resolve an issue in a repository efficiently and decisively.\n"
        "Use the provided tools (search_similar_code, read_file, edit_file, run_command, submit_patch) to localize, repair, verify, and submit your patch.\n\n"
        f"Repository: {repo_path.name}\n"
        f"Issue Description:\n{task_description}\n"
    )
    dialogue: List[str] = [system_prompt]

    for turn in range(1, max_turns + 1):
        console.print(f"\n[bold green]━━━━━━━━━━━━━━━━ Turn {turn}/{max_turns} ━━━━━━━━━━━━━━━━[/bold green]")
        prompt_text = "\n".join(dialogue) + "\n<action>\n"
        response = generate_fn(prompt_text)
        clean_resp = response.split("</action>")[0].strip()

        tool_call = MultiTurnAgent.parse_tool_call(clean_resp)
        if tool_call:
            tname, targs = tool_call
            console.print(f"[bold cyan]Agent Action:[/bold cyan] {tname}({list(targs.keys())})")

            # In interactive mode, ask user confirmation before modifying or running
            if interactive and tname in ("edit_file", "run_command"):
                if not Confirm.ask(f"Allow agent to execute [bold]{tname}[/bold] with args {targs}?"):
                    console.print("[yellow]Action aborted by user.[/yellow]")
                    dialogue.append(f"<action>\n{clean_resp}\n</action>")
                    dialogue.append("<observation>\nUser rejected action.\n</observation>")
                    continue

            if tname == "read_file":
                obs = harness.read_file(targs.get("filepath") or targs.get("path", ""), int(targs.get("start_line", 1)), int(targs.get("end_line", 60)))
            elif tname == "edit_file":
                obs = harness.edit_file(targs.get("filepath") or targs.get("path", ""), targs.get("old_string", ""), targs.get("new_string", ""))
            elif tname == "run_command":
                obs = harness.run_command(targs.get("command") or targs.get("cmd", ""))
            elif tname == "search_similar_code":
                obs = harness.search_similar_code(targs.get("query", ""))
            elif tname == "submit_patch":
                console.print("[bold green]Agent completed task and called submit_patch![/bold green]")
                break
            else:
                obs = json.dumps({"status": "error", "error": f"Unknown tool '{tname}'"})

            console.print(f"[dim]Observation:[/dim] {obs[:300]}...")
            dialogue.append(f"<action>\n{clean_resp}\n</action>")
            dialogue.append(f"<observation>\n{obs[:500]}\n</observation>")
        else:
            # Check for SRI format blocks
            sri_blocks = SRIFormatter.parse(clean_resp)
            if sri_blocks:
                console.print(f"[bold green]Detected {len(sri_blocks)} SRI patch block(s)![/bold green]")
                for b in sri_blocks:
                    harness.edit_file(b.file_path, b.search_content, b.replace_content)
                break
            if "submit_patch" in clean_resp:
                console.print("[bold green]Agent finished and concluded task.[/bold green]")
                break
            dialogue.append(f"<action>\n{clean_resp}\n</action>")

    # 4. Display Final Git Diff
    diff = harness.get_diff()
    if diff.strip():
        console.print("\n[bold green]Proposed Changes (Git Diff):[/bold green]")
        console.print(Syntax(diff, "diff", theme="monokai", line_numbers=True))

        if dry_run:
            console.print("[yellow]Dry-run mode active: rolling back all changes...[/yellow]")
            harness.rollback()
            console.print("[green]Workspace cleanly restored.[/green]")
        elif interactive:
            if not Confirm.ask("Keep and commit these changes?"):
                harness.rollback()
                console.print("[yellow]Changes reverted.[/yellow]")
            else:
                console.print("[bold green]Changes preserved in repository![/bold green]")
    else:
        console.print("[yellow]No code modifications were produced by the agent.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous SWE Coding Agent CLI")
    parser.add_argument("--repo", type=str, default=".", help="Path to repository or project directory")
    parser.add_argument("--task", type=str, default=None, help="Task description or bug report")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rlvr_competition/adapter_model.pt", help="Path to DR-LoRA adapter weights")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns for agent exploration")
    parser.add_argument("--interactive", action="store_true", help="Interactive turn approval mode")
    parser.add_argument("--dry-run", action="store_true", help="Preview proposed diff without writing modifications")
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    if not repo_dir.exists() or not repo_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Repository path '{args.repo}' does not exist.")
        sys.exit(1)

    task_desc = args.task
    if not task_desc:
        console.print(f"[bold cyan]Target Repository:[/bold cyan] {repo_dir}")
        task_desc = Prompt.ask("[bold green]What coding task or bug should the agent resolve?[/bold green]")

    run_agent_session(
        repo_path=repo_dir,
        task_description=task_desc,
        checkpoint=args.checkpoint,
        max_turns=args.max_turns,
        interactive=args.interactive,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
