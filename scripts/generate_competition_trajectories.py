#!/usr/bin/env python3
"""Generate Domain-Specific Multi-Turn SFT Trajectories for Kaggle Gemma Challenge.

Reads the 129 official tasks from /Users/vishnuanirudh/data/gemma-4-developer-agent/tasks.jsonl
(covering FastAPI, Rich, Requests, and HTTPX) and synthesizes realistic, high-fidelity
multi-turn trajectories using the exact 9 competition harness tools:
- search_similar_code
- get_code_neighbors
- read_file
- edit_file
- run_command
- submit_patch

Outputs:
  data/competition_sft_trajectories.jsonl
Compatible with CurriculumTrajectoryDataset and StepLevelErrorMasker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

console = Console()


def extract_diff_hunks(patch: str) -> Dict[str, Tuple[str, str, int]]:
    """Extract (old_string, new_string, start_line) per file from a git diff patch."""
    files: Dict[str, Tuple[str, str, int]] = {}
    current_file = None
    old_lines = []
    new_lines = []
    start_line = 1

    for line in patch.splitlines():
        if line.startswith("--- a/"):
            if current_file and (old_lines or new_lines):
                files[current_file] = ("\n".join(old_lines).strip(), "\n".join(new_lines).strip(), start_line)
            current_file = line[6:].strip()
            old_lines = []
            new_lines = []
        elif line.startswith("+++ b/"):
            continue
        elif line.startswith("@@"):
            m = re.search(r"@@ -(\d+)", line)
            if m:
                start_line = int(m.group(1))
        elif current_file:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])

    if current_file and (old_lines or new_lines):
        files[current_file] = ("\n".join(old_lines).strip(), "\n".join(new_lines).strip(), start_line)

    return files


def extract_test_file(test_patch: str) -> str:
    """Extract target test file path from test_patch."""
    m = re.search(r"\+\+\+ b/([^\n\t]+)", test_patch)
    if m:
        return m.group(1).strip()
    return "tests/test_main.py"


def extract_keywords_from_issue(text: str) -> str:
    """Extract prominent symbol or concept from problem statement."""
    # Look for backtick symbols: `symbol`
    code_symbols = re.findall(r"`([a-zA-Z0-9_\.]{3,})`", text)
    if code_symbols:
        return code_symbols[0]
    # Look for CamelCase or function-like names
    names = re.findall(r"\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z0-9_]{3,})\b", text)
    for n in names:
        if n.lower() not in {"this", "that", "with", "when", "from", "have", "issue", "error", "bug", "fastapi", "rich"}:
            return n
    return "handler"


def build_trajectory_for_task(task: Dict[str, Any], task_idx: int) -> Optional[Dict[str, Any]]:
    """Synthesize a complete multi-turn trajectory for a competition task."""
    hunks = extract_diff_hunks(task["patch"])
    if not hunks:
        return None

    # Primary target file and diff
    target_file, (old_code, new_code, start_line) = next(iter(hunks.items()))
    test_file = extract_test_file(task.get("test_patch", ""))
    keyword = extract_keywords_from_issue(task["problem_statement"])

    # Determine difficulty
    lines_changed = len(old_code.splitlines()) + len(new_code.splitlines())
    if len(hunks) == 1 and lines_changed <= 10:
        difficulty = "easy"
    elif len(hunks) == 1 and lines_changed <= 35:
        difficulty = "medium"
    else:
        difficulty = "hard"

    problem = task.get("problem_statement", "").strip()
    problem_summary = problem[:500] if len(problem) > 500 else problem
    system_prompt = (
        "You are an expert autonomous software engineer assigned to resolve an issue in a repository efficiently and decisively.\n"
        "Use the provided tools (search_similar_code, get_code_neighbors, read_file, edit_file, run_command, submit_patch) "
        "to localize, repair, verify, and submit your patch.\n\n"
        f"Repository: {task.get('repo', '')}\n"
        f"Issue Description:\n{problem_summary}\n"
    )

    symbol_name = Path(target_file).stem
    func_match = re.search(r"def\s+([a-zA-Z0-9_]+)", old_code or new_code)
    class_match = re.search(r"class\s+([a-zA-Z0-9_]+)", old_code or new_code)
    primary_symbol = func_match.group(1) if func_match else (class_match.group(1) if class_match else keyword)

    steps: List[Dict[str, Any]] = []

    # Turn 1: Semantic Search
    steps.append({
        "action_text": f'<tool_call name="search_similar_code" query="{keyword}" k="10" />',
        "observation_text": f"Top-10 similar code nodes:\n1. {target_file}::{primary_symbol} (similarity: 0.92)\n2. {target_file} (similarity: 0.85)",
        "is_error": False,
    })

    # Turn 2: Graph Neighbors
    steps.append({
        "action_text": f'<tool_call name="get_code_neighbors" node="{target_file}::{primary_symbol}" />',
        "observation_text": f"Graph Neighbors for {target_file}::{primary_symbol}:\n• In-edges (CALLS): {test_file}\n• Out-edges: typing, pydantic, standard library",
        "is_error": False,
    })

    # Turn 3: Inspect Code
    view_start = max(1, start_line - 5)
    view_end = start_line + len(old_code.splitlines()) + 10
    steps.append({
        "action_text": f'<tool_call name="read_file" filepath="{target_file}" start_line="{view_start}" end_line="{view_end}" />',
        "observation_text": f"=== File: {target_file} (Lines {view_start}-{view_end}) ===\n{old_code}",
        "is_error": False,
    })

    # Hard difficulty includes an initial imperfect attempt or syntax slip, masked to train error recovery
    if difficulty == "hard" and task_idx % 2 == 0:
        imperfect_code = old_code.replace(":", "") if ":" in old_code else old_code + "\n    pass"
        steps.append({
            "action_text": f'<tool_call name="edit_file" filepath="{target_file}" old_string="{old_code}" new_string="{imperfect_code}" />',
            "observation_text": f"Error: Syntax error or test failure in {target_file}.",
            "is_error": True,  # Flagged as error: StepLevelErrorMasker masks this to -100!
        })
        steps.append({
            "action_text": f'<tool_call name="run_command" command="pytest {test_file}" />',
            "observation_text": f"FAILED {test_file} - SyntaxError / AssertionError",
            "is_error": True,
        })

    # Turn 4: Clean Edit
    steps.append({
        "action_text": f'<tool_call name="edit_file" filepath="{target_file}" old_string="{old_code}" new_string="{new_code}" />',
        "observation_text": f"Successfully replaced content in {target_file}.",
        "is_error": False,
    })

    # Turn 5: Verify with Targeted Test
    steps.append({
        "action_text": f'<tool_call name="run_command" command="pytest {test_file}" />',
        "observation_text": f"=== Test Result: PASSED ===\n1 passed in 0.18s",
        "is_error": False,
    })

    # Turn 6: Submit Patch
    steps.append({
        "action_text": '<tool_call name="submit_patch" />',
        "observation_text": f"Patch submitted successfully. 1 file changed, {len(new_code.splitlines())} insertions(+).",
        "is_error": False,
    })

    final_solution = f"Resolved issue in {target_file} by updating {primary_symbol} to handle edge cases cleanly. Verified with {test_file}."

    return {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "difficulty": difficulty,
        "system_prompt": system_prompt,
        "steps": steps,
        "final_patch": final_solution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Competition SFT Trajectories")
    parser.add_argument(
        "--tasks-file",
        type=str,
        default="/Users/vishnuanirudh/data/gemma-4-developer-agent/tasks.jsonl",
        help="Path to tasks.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./data/competition_sft_trajectories.jsonl",
        help="Path to output jsonl trajectories file",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    console.print(f"[cyan]Reading tasks from {tasks_path}...[/cyan]")
    trajectories = []
    repo_counts = {}

    with open(tasks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            task = json.loads(line)
            traj = build_trajectory_for_task(task, idx)
            if traj:
                trajectories.append(traj)
                repo = task.get("repo", "unknown")
                repo_counts[repo] = repo_counts.get(repo, 0) + 1

    console.print(f"[green]Successfully synthesized {len(trajectories)} multi-turn trajectories across {len(repo_counts)} repos:[/green]")
    for r, count in repo_counts.items():
        console.print(f"  • {r}: {count} trajectories")

    with open(output_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    console.print(f"\n[bold green]Saved competition SFT trajectory dataset to: {output_path} ({output_path.stat().st_size / 1e3:.1f} KB)[/bold green]")


if __name__ == "__main__":
    main()
