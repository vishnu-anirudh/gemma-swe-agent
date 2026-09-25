"""Repository Inspection and Execution Tools for Autonomous SWE Agent."""

from __future__ import annotations

import os
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

            selected = [
                f"{i + 1:4d} | {lines[i]}" for i in range(start_idx, end_idx)
            ]
            header = f"=== File: {file_path} (Showing lines {start_idx + 1}-{end_idx} of {total_lines}) ===\n"
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
        try:
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
            if not lines:
                return f"No matches found for '{query}'."

            # Filter out hidden and virtualenv files
            filtered = [
                line for line in lines if not any(x in line for x in [".git", "__pycache__", ".venv"])
            ]

            def rank_match(line: str) -> int:
                parts = line.split(":", 2)
                path = parts[0] if len(parts) > 0 else ""
                content = parts[2] if len(parts) > 2 else ""

                score = 0
                # Top priority: Function or class definitions
                if any(kw in content for kw in ["def ", "class ", "__all__"]):
                    score += 100
                # High priority: Core python source code
                if path.endswith(".py"):
                    if "/tests/" not in path and not path.startswith("./tests"):
                        score += 50
                    else:
                        score += 20
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
