"""Repository Inspection and Execution Tools for Autonomous SWE Agent."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class RepoTools:
    """Safe, read-only and test execution tools operating on a target repository."""

    @classmethod
    def view_file(
        cls, repo_dir: str, file_path: str, start_line: int = 1, end_line: int = 100
    ) -> str:
        """View lines from a file in the repository (1-indexed)."""
        full_path = Path(repo_dir) / file_path
        if not full_path.exists():
            return f"Error: File '{file_path}' does not exist in repository."

        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total_lines = len(lines)
            start_idx = max(0, start_line - 1)
            max_window = 60
            end_idx = min(total_lines, min(end_line, start_idx + max_window))

            selected = [lines[i] for i in range(start_idx, end_idx)]
            header = f"=== File: {file_path} (Lines {start_idx + 1}-{end_idx} of {total_lines}) ===\n"
            return header + "\n".join(selected)
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"

    @classmethod
    def list_dir(cls, repo_dir: str, dir_path: str = ".") -> str:
        """List files and directories in the repository."""
        target = Path(repo_dir) / dir_path
        if not target.exists() or not target.is_dir():
            return f"Error: Directory '{dir_path}' not found."

        entries = []
        try:
            for item in sorted(target.iterdir()):
                if item.name.startswith((".", "__pycache__")):
                    continue
                kind = "DIR " if item.is_dir() else "FILE"
                entries.append(f"[{kind}] {item.name}")
            return "\n".join(entries[:60])  # Cap at 60 items
        except Exception as e:
            return f"Error listing directory '{dir_path}': {e}"

    @classmethod
    def search_code(cls, repo_dir: str, query: str, max_results: int = 15) -> str:
        """Search code files in the repository for a query string or regex, ranking definitions and source files first."""
        if not query or not query.strip():
            return "Error: query cannot be empty. Please provide a symbol or text to search."
        try:
            is_git = (Path(repo_dir) / ".git").exists() or (Path(repo_dir) / ".git").is_file()
            lines = []

            # 1. Direct search via git grep or grep
            if is_git:
                proc = subprocess.run(
                    ["git", "grep", "-n", "-I", "-i", "-F", query],
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0:
                    lines = proc.stdout.splitlines()

            if not lines:
                cmd = ["grep", "-rnI", query, "."]
                proc = subprocess.run(
                    cmd,
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                lines = proc.stdout.splitlines()

            # 2. Expand dotted queries and CamelCase / identifier tokens from multi-word queries
            candidates = []
            if "." in query and not query.endswith(".py"):
                candidates.append(query.split(".")[-1].strip())

            # Extract distinct CamelCase and identifier tokens (e.g. FilePathField from 'FilePathField path=os.path.join')
            tokens = re.findall(r"[A-Z][a-zA-Z0-9]+|[a-zA-Z_]\w{3,}", query)
            ignored_tokens = {"path", "self", "file", "join", "none", "true", "false", "test", "tests"}
            for tok in tokens:
                if tok not in candidates and len(tok) >= 3 and tok.lower() not in ignored_tokens:
                    candidates.append(tok)

            for symbol in candidates[:3]:
                for prefix in [f"class {symbol}", f"def {symbol}", symbol]:
                    if is_git:
                        proc_sub = subprocess.run(
                            ["git", "grep", "-n", "-I", prefix, "--", "*.py"],
                            cwd=repo_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=5,
                        )
                        if proc_sub.returncode == 0:
                            lines = proc_sub.stdout.splitlines() + lines
                    else:
                        proc_sub = subprocess.run(
                            ["grep", "-rnI", "--include=*.py", prefix, "."],
                            cwd=repo_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=5,
                        )
                        lines = proc_sub.stdout.splitlines() + lines

            if not lines:
                return f"No matches found for '{query}'."

            # Filter out hidden and virtualenv files and deduplicate
            filtered = [
                line for line in lines if not any(x in line for x in [".git", "__pycache__", ".venv"])
            ]
            filtered = list(dict.fromkeys(filtered))

            def rank_match(line: str) -> int:
                parts = line.split(":", 2)
                path = parts[0] if len(parts) > 0 else ""
                content = parts[2].strip() if len(parts) > 2 else ""

                score = 0
                is_test = (
                    path.startswith("tests/")
                    or path.startswith("./tests")
                    or "/tests/" in path
                    or "/test_" in path
                    or path.startswith("test_")
                )
                # Top priority: Function, class, or constant/setting definitions
                if any( content.startswith(kw) or f" {kw}" in content for kw in ["def ", "class ", "__all__"]):
                    score += 100
                elif re.match(r"^[A-Z_][A-Z0-9_]*\s*=", content):
                    score += 110

                # High priority: Core non-test python source code
                if path.endswith(".py"):
                    if not is_test:
                        score += 80
                    else:
                        score -= 40
                # Lower priority: Documentation and galleries
                elif any(d in path for d in ["/docs/", ".rst", ".md", ".txt"]):
                    score -= 30
                return -score

            sorted_lines = sorted(filtered, key=rank_match)
            limited = sorted_lines[:max_results]
            return "\n".join(limited) + (
                f"\n... ({len(filtered) - max_results} more matches truncated)"
                if len(filtered) > max_results
                else ""
            )
        except subprocess.TimeoutExpired:
            return "Search timed out."
        except Exception as e:
            return f"Search error: {e}"

    @classmethod
    def run_test(cls, repo_dir: str, test_cmd: str, timeout: int = 30) -> str:
        """Run pytest or test commands inside the repository sandbox."""
        # Sanity check against malicious/destructive commands
        blocked = ["rm -rf", "git reset", "curl", "wget", "nc", "bash -i"]
        if any(b in test_cmd for b in blocked):
            return "Error: Command contains blocked operations."

        try:
            proc = subprocess.run(
                test_cmd,
                shell=True,
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            output = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
            status = "PASSED" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
            # Return last 40 lines of test output to keep context manageable
            truncated = "\n".join(output.splitlines()[-40:])
            return f"=== Test Result: {status} ===\n{truncated}"
        except subprocess.TimeoutExpired:
            return f"Error: Test execution timed out after {timeout} seconds."
        except Exception as e:
            return f"Execution error: {e}"

    @classmethod
    def get_graph_context(cls, repo_dir: str, query: str, max_symbols: int = 5) -> str:
        """Query the AST code graph for symbols, callers, callees, and definition locations."""
        if not query or not query.strip():
            return "Error: query cannot be empty. Please provide a symbol or identifier to look up."

        try:
            from src.doctor.graph_engine import CodeGraphEngine

            engine = CodeGraphEngine()
            repo_path = Path(repo_dir)

            # Discover Python files in repo (prioritizing src/ or package directories)
            candidates: List[Path] = []
            for sub in ["src", "lib", "."]:
                target_dir = repo_path / sub
                if target_dir.exists():
                    for p in target_dir.rglob("*.py"):
                        parts = p.parts
                        if not any(part.startswith((".", "__pycache__", "tests", "docs")) for part in parts):
                            candidates.append(p)
                        if len(candidates) >= 80:
                            break
                if candidates:
                    break

            for cf in candidates:
                rel = str(cf.relative_to(repo_path)).replace("\\", "/")
                try:
                    engine.add_file(rel, cf.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass

            # Search nodes matching query
            clean_q = query.lower().strip()
            matches = []
            for nid, node in engine.nodes.items():
                if node.node_type == "module":
                    continue
                score = 0
                node_lower = node.name.lower()
                if clean_q == node_lower:
                    score += 10
                elif clean_q in node_lower:
                    score += 5
                elif clean_q in node.file_path.lower():
                    score += 2

                if score > 0:
                    matches.append((score, node))

            matches.sort(key=lambda x: -x[0])
            if not matches:
                return f"No code-graph symbols matching '{query}' found."

            lines = [f"=== Code Graph Context for '{query}' ==="]
            for _, node in matches[:max_symbols]:
                callers = engine.get_callers(node.id)
                callees = engine.get_callees(node.id)
                caller_str = ", ".join(callers[:3]) if callers else "none"
                callee_str = ", ".join(callees[:3]) if callees else "none"
                sig = f"def {node.name}{node.signature}" if node.node_type in ("function", "method") else f"class {node.name}"
                lines.append(
                    f"• [{node.node_type.upper()}] {node.id} (Lines {node.line_start}-{node.line_end})\n"
                    f"  File: {node.file_path}\n"
                    f"  Signature: {sig}\n"
                    f"  Callers ({len(callers)}): {caller_str}\n"
                    f"  Callees ({len(callees)}): {callee_str}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Code graph query note: {e}"
