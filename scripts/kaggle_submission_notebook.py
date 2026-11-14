#!/usr/bin/env python3
"""Kaggle Competition Submission Script & Standalone Runner.

Designed for the Kaggle Gemma Challenge:
1. Discovers inputs from /kaggle/input/gemma-4-developer-agent (or local fallback).
2. Unpacks submission.zip / adapters / prompts.
3. Initializes the post-trained Gemma agent with DR-LoRA adapters.
4. Executes autonomous multi-turn solving on competition test instances.
5. Saves results to /kaggle/working/submission.parquet (and submission.csv fallback).
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
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

console = Console()


def find_input_paths() -> Tuple[Path, Path, Path, Path]:
    """Locate task definitions, snapshots, graphs, and submission assets across Kaggle and local environments."""
    kaggle_data_dir = Path("/kaggle/input/gemma-4-developer-agent")
    local_data_dir = Path("/Users/vishnuanirudh/data/gemma-4-developer-agent")

    data_dir = kaggle_data_dir if kaggle_data_dir.exists() else local_data_dir
    if not data_dir.exists():
        data_dir = Path("./data")

    tasks_path = data_dir / "tasks.jsonl"
    snapshots_dir = data_dir / "snapshots"
    graphs_dir = data_dir / "graphs"

    # Submission bundle search paths
    sub_paths = [
        Path("/kaggle/working/submission"),
        Path("/kaggle/input/gemma-submission/submission"),
        Path("/kaggle/input/gemma-submission/submission.zip"),
        Path("./submission"),
        Path("./submission.zip"),
    ]

    submission_root = None
    for p in sub_paths:
        if p.exists():
            if p.is_file() and p.suffix == ".zip":
                extract_target = Path("/tmp/kaggle_unpacked_submission")
                extract_target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(extract_target)
                submission_root = extract_target
                break
            elif p.is_dir():
                submission_root = p
                break

    if not submission_root:
        submission_root = Path("./submission")

    return tasks_path, snapshots_dir, graphs_dir, submission_root


class SandboxedHarness:
    """Implements the 9 official competition tools within an isolated workspace."""

    def __init__(
        self,
        workspace_dir: Path,
        graph_data: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 15,
    ):
        self.workspace_dir = workspace_dir
        self.graph_data = graph_data or {"nodes": {}, "edges": []}
        self.max_tool_calls = max_tool_calls
        self.tool_calls_used = 0
        self.patch_submitted = False
        self.submitted_patch = ""

    def _resolve_file_path(self, filepath: str) -> Optional[Path]:
        clean = filepath.strip().lstrip("/")
        target = self.workspace_dir / clean
        if target.exists() and target.is_file():
            return target

        # Candidate 1: dot-path to slash path + .py
        as_path = clean.replace(".", "/")
        cand = self.workspace_dir / f"{as_path}.py"
        if cand.exists() and cand.is_file():
            return cand

        # Candidate 2: strip trailing symbol from dot-path
        parts = clean.split(".")
        for i in range(len(parts) - 1, 0, -1):
            sub_path = "/".join(parts[:i])
            cand = self.workspace_dir / f"{sub_path}.py"
            if cand.exists() and cand.is_file():
                return cand

        return None

    def read_file(self, filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        self.tool_calls_used += 1
        target = self._resolve_file_path(filepath)
        if not target:
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found"})

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            s_idx = max(0, (start_line or 1) - 1)
            e_idx = min(total, (end_line or (s_idx + 120)))
            content = "\n".join(lines[s_idx:e_idx])
            return json.dumps({
                "status": "ok",
                "filepath": str(target.relative_to(self.workspace_dir)),
                "content": content,
                "start_line": s_idx + 1,
                "end_line": e_idx,
                "total_lines": total,
            })
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        self.tool_calls_used += 1
        target = self._resolve_file_path(filepath)
        if not target:
            return json.dumps({"status": "error", "error": f"File '{filepath}' not found"})

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            occurrences = content.count(old_string)
            if occurrences == 0:
                # Flexible fallback: strip trailing whitespace per line
                old_lines = [l.strip() for l in old_string.splitlines() if l.strip()]
                target_lines = content.splitlines()
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
                    target.write_text(new_content, encoding="utf-8")
                    return json.dumps({"status": "ok", "filepath": str(target.relative_to(self.workspace_dir)), "strategy": "flexible_line_match"})
                return json.dumps({"status": "error", "error": "old_string pattern not found in file"})

            if occurrences > 1 and not allow_multiple:
                return json.dumps({"status": "error", "error": f"Multiple occurrences ({occurrences}) found without allow_multiple=True"})

            new_content = content.replace(old_string, new_string)
            target.write_text(new_content, encoding="utf-8")
            return json.dumps({"status": "ok", "filepath": str(target.relative_to(self.workspace_dir)), "strategy": "exact", "occurrences": occurrences})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def write_file(self, filepath: str, content: str) -> str:
        self.tool_calls_used += 1
        target = self._resolve_file_path(filepath) or (self.workspace_dir / filepath.lstrip("/"))
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
        return json.dumps({
            "status": "ok",
            "tool_calls_used": self.tool_calls_used,
            "max_tool_calls": self.max_tool_calls,
            "patch_submitted": self.patch_submitted,
            "patch_size": len(self.submitted_patch),
        })

    def submit_patch(self) -> str:
        try:
            subprocess.run(["git", "add", "-N", "."], cwd=str(self.workspace_dir), capture_output=True)
            proc = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            raw_patch = proc.stdout
            # Strip protected test files from patch to comply with Phase 2 verification rules
            clean_patch = self._strip_protected_files(raw_patch)
            self.submitted_patch = clean_patch
            self.patch_submitted = True
            return json.dumps({
                "status": "ok",
                "patch_size": len(self.submitted_patch),
                "is_empty": len(self.submitted_patch.strip()) == 0,
            })
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _strip_protected_files(self, patch: str) -> str:
        """Strip changes to test files and runner configs from the patch."""
        clean_lines = []
        skipping = False
        for line in patch.splitlines(keepends=True):
            if line.startswith("diff --git"):
                parts = line.split()
                target_path = parts[-1].lstrip("b/") if len(parts) >= 4 else ""
                is_test = (
                    target_path.startswith("tests/")
                    or target_path.startswith("test/")
                    or "test_" in target_path
                    or "_test.py" in target_path
                    or target_path in ("conftest.py", "pytest.ini", "pyproject.toml")
                )
                skipping = is_test
            if not skipping:
                clean_lines.append(line)
        return "".join(clean_lines)

    def _get_normalized_nodes(self) -> Dict[str, Dict[str, Any]]:
        raw_nodes = self.graph_data.get("nodes", {})
        if isinstance(raw_nodes, dict):
            return raw_nodes
        elif isinstance(raw_nodes, list):
            norm = {}
            for item in raw_nodes:
                if isinstance(item, dict):
                    nid = item.get("id") or item.get("name") or str(len(norm))
                    norm[nid] = item
            return norm
        return {}

    def get_code_neighbors(self, node: str, edge_type: Optional[str] = None, max_neighbors: int = 20) -> str:
        self.tool_calls_used += 1
        nodes = self._get_normalized_nodes()
        edges = self.graph_data.get("edges", [])

        matched_id = None
        for nid in nodes:
            if node == nid or node.lower() in nid.lower():
                matched_id = nid
                break

        if not matched_id:
            return json.dumps({"status": "ok", "node": node, "neighbors": [], "count": 0})

        neighbors = []
        for e in edges:
            src = e.get("source") or e.get("from")
            dst = e.get("target") or e.get("to")
            if src == matched_id or dst == matched_id:
                other = dst if src == matched_id else src
                if edge_type and e.get("type", "").lower() != edge_type.lower():
                    continue
                if other:
                    neighbors.append(other)

        neighbors = list(dict.fromkeys(neighbors))[:max_neighbors]
        return json.dumps({"status": "ok", "node": matched_id, "neighbors": neighbors, "count": len(neighbors)})

    def search_similar_code(self, query: str, k: int = 5) -> str:
        self.tool_calls_used += 1
        nodes = self._get_normalized_nodes()
        results = []
        clean_q = query.lower().strip()

        for nid, ndata in nodes.items():
            name = ndata.get("name", "")
            code_snip = (ndata.get("text") or ndata.get("code") or "")[:200]
            score = 0.0
            if clean_q == name.lower() or clean_q == nid.lower():
                score = 1.0
            elif clean_q in name.lower() or clean_q in nid.lower():
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
        sub_edges = []
        for e in all_edges:
            src = e.get("source") or e.get("from")
            dst = e.get("target") or e.get("to")
            if src in node_set and dst in node_set:
                sub_edges.append(e)

        return json.dumps({
            "status": "ok",
            "nodes": list(node_set),
            "edges": sub_edges,
            "node_count": len(node_set),
            "edge_count": len(sub_edges),
        })


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse tool calls across XML, function calls, and markdown styles."""
    known_keys = [
        "name", "filepath", "path", "old_string", "new_string", "query",
        "node", "node_name", "command", "cmd", "k", "start_line",
        "end_line", "patch", "allow_multiple"
    ]
    key_pattern = "|".join(known_keys)

    # 1. XML style: <tool_call name="..." ... />
    xml_match = re.search(r'<tool_call\s+name="([^"]+)"([^>]*)/?>', text, re.DOTALL)
    if xml_match:
        tool_name = xml_match.group(1).strip()
        attrs_str = xml_match.group(2)
        pattern = rf'({key_pattern})\s*=\s*(?:\"(.*?)\"|\x27(.*?)\x27)(?=\s*(?:,\s*)?(?:{key_pattern})\s*=|\s*/>|\s*$)'
        matches = re.findall(pattern, attrs_str, re.DOTALL)
        attrs = {}
        for k, v1, v2 in matches:
            val = v1 if v1 != "" else v2
            attrs[k] = val.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n')
        if not attrs:
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))
        return tool_name, attrs

    # 2. Block XML style: <tool>name</tool>
    xml_block = re.search(r'<tool>([a-zA-Z0-9_]+)</tool>(.*?)(?:</action>|<action>|$)', text, re.DOTALL)
    if xml_block:
        tool_name = xml_block.group(1).strip()
        body = xml_block.group(2)
        attrs = {}
        for tag in ["query", "node", "filepath", "path", "start_line", "end_line", "command", "cmd", "old_string", "new_string", "content", "k"]:
            val_m = re.search(rf'<{tag}>([^<]*)</{tag}>', body, re.DOTALL)
            if val_m:
                attrs[tag] = val_m.group(1).strip()
        return tool_name, attrs

    return None


def execute_agent_loop(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    task: Dict[str, Any],
    snapshots_dir: Path,
    graphs_dir: Path,
    max_turns: int = 6,
) -> str:
    """Run full multi-turn solving loop on a single task and return generated git diff patch."""
    iid = task["instance_id"]
    repo = task.get("repo", "")
    commit = task.get("base_commit", "")
    problem = task.get("problem_statement", "").strip()

    # Workspace setup
    workspace = Path(f"/tmp/kaggle_workspaces/{iid}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    snapshot_tar = snapshots_dir / f"{iid}.tgz"
    if snapshot_tar.exists():
        with tarfile.open(snapshot_tar, "r:gz") as tar:
            tar.extractall(path=workspace)
    else:
        subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
        (workspace / "README.md").write_text(f"# {repo}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(workspace), capture_output=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=str(workspace), capture_output=True)

    # Load AST graph
    graph_path = graphs_dir / f"{repo.split('/')[-1]}_{commit}.json"
    graph_data = None
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as gf:
                graph_data = json.load(gf)
        except Exception:
            pass

    harness = SandboxedHarness(workspace_dir=workspace, graph_data=graph_data)

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

    system_prompt = (
        "You are an expert autonomous software engineer assigned to resolve an issue in a repository efficiently and decisively.\n"
        "Use the provided tools (search_similar_code, get_code_neighbors, read_file, edit_file, run_command, submit_patch) to localize, repair, verify, and submit your patch.\n\n"
        f"Repository: {repo}\n"
        f"Issue Description:\n{problem[:500]}\n"
    )
    dialogue: List[str] = [system_prompt]

    for turn in range(1, max_turns + 1):
        prompt_text = "\n".join(dialogue) + "\n<action>\n"
        response = generate_fn(prompt_text)
        clean_resp = response.split("</action>")[0].strip()

        tool_call = parse_tool_call(clean_resp)
        if tool_call:
            tname, targs = tool_call
            if tname == "read_file":
                obs = harness.read_file(targs.get("filepath") or targs.get("path", ""), int(targs.get("start_line", 1)), int(targs.get("end_line", 50)))
            elif tname == "edit_file":
                obs = harness.edit_file(targs.get("filepath") or targs.get("path", ""), targs.get("old_string", ""), targs.get("new_string", ""))
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

    # Workspace cleanup to stay within Kaggle disk limits
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)

    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Inference & Submission Generator")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to evaluate")
    parser.add_argument("--output-parquet", type=str, default="/kaggle/working/submission.parquet", help="Output parquet path")
    parser.add_argument("--output-csv", type=str, default="/kaggle/working/submission.csv", help="Output csv path")
    parser.add_argument("--max-turns", type=int, default=6, help="Max agent turns per task")
    args = parser.parse_args()

    # Determine runtime working directory
    working_dir = Path("/kaggle/working")
    if not working_dir.exists():
        working_dir = Path("./results")
    working_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = Path(args.output_parquet) if "/kaggle/working" in str(args.output_parquet) and Path("/kaggle/working").exists() else working_dir / "submission.parquet"
    out_csv = Path(args.output_csv) if "/kaggle/working" in str(args.output_csv) and Path("/kaggle/working").exists() else working_dir / "submission.csv"

    tasks_path, snapshots_dir, graphs_dir, sub_root = find_input_paths()

    console.print(f"[bold cyan]Kaggle Inference Environment:[/bold cyan]")
    console.print(f"  • Tasks Path: {tasks_path}")
    console.print(f"  • Snapshots Dir: {snapshots_dir}")
    console.print(f"  • Graphs Dir: {graphs_dir}")
    console.print(f"  • Submission Root: {sub_root}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"[cyan]Using compute device: {device}[/cyan]")

    # Load Base Model & Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "google/gemma-2-2b-it"
    console.print(f"[cyan]Loading base model '{base_model_name}'...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    # Attach LoRA adapter if present
    adapter_dir = sub_root / "adapters" / "main_lora"
    if (adapter_dir / "adapter_config.json").exists():
        console.print(f"[green]Attaching fine-tuned PEFT adapter from {adapter_dir}...[/green]")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # Load tasks
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    if args.limit:
        tasks = tasks[: args.limit]

    console.print(f"[bold green]Starting Inference on {len(tasks)} tasks...[/bold green]\n")

    results = []
    start_time = time.time()

    for idx, t in enumerate(tasks):
        iid = t["instance_id"]
        console.print(f"[bold]Task {idx + 1}/{len(tasks)}:[/bold] {iid} ({t.get('repo', '')})")
        t0 = time.time()
        patch = execute_agent_loop(
            model=model,
            tokenizer=tokenizer,
            device=device,
            task=t,
            snapshots_dir=snapshots_dir,
            graphs_dir=graphs_dir,
            max_turns=args.max_turns,
        )
        elapsed = time.time() - t0
        has_patch = len(patch.strip()) > 0
        console.print(f"  -> Generated patch: {len(patch)} bytes (took {elapsed:.1f}s)")
        results.append({
            "id": iid,
            "prediction": patch,
        })

    total_time = time.time() - start_time
    df = pd.DataFrame(results)

    # Save Parquet & CSV
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    console.print(f"\n[bold green]Successfully generated submissions:[/bold green]")
    console.print(f"  • Parquet: {out_parquet} ({out_parquet.stat().st_size / 1e3:.1f} KB)")
    console.print(f"  • CSV: {out_csv} ({out_csv.stat().st_size / 1e3:.1f} KB)")
    console.print(f"  • Total Time: {total_time:.1f}s | Avg: {total_time / len(tasks):.1f}s/task")


if __name__ == "__main__":
    main()
