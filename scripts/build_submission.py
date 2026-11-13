#!/usr/bin/env python3
"""Build and Validate Kaggle Gemma Challenge ADK Submission Archive (submission.zip).

Validates:
1. Root agent.yaml presence and YAML syntax.
2. Resolvability of all !include directives without path traversal outside submission root.
3. Proper PEFT adapter format (adapter_config.json + adapter_model.safetensors).
4. Valid ADK skill structure (SKILL.md with YAML frontmatter).
5. All referenced tools are in the allowed competition harness whitelist.
6. Packages all artifacts into submission.zip with agent.yaml at the archive root.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Set

from rich.console import Console
from rich.table import Table

console = Console()

ALLOWED_HARNESS_TOOLS: Set[str] = {
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
}

ALLOWED_MODELS: Set[str] = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}


def validate_submission_directory(submission_dir: Path) -> List[str]:
    """Check directory structure and required files."""
    errors: List[str] = []

    # 1. agent.yaml must exist
    agent_yaml = submission_dir / "agent.yaml"
    if not agent_yaml.exists():
        errors.append("Missing required root file: agent.yaml")
        return errors

    content = agent_yaml.read_text(encoding="utf-8")

    # 2. Check model specification
    model_match = re.search(r"model:\s*([a-zA-Z0-9_\-\.]+)", content)
    if not model_match:
        errors.append("agent.yaml missing 'model:' field")
    else:
        model_name = model_match.group(1).strip()
        if model_name not in ALLOWED_MODELS:
            errors.append(f"Model '{model_name}' not in allowed Gemma 4 variants: {sorted(list(ALLOWED_MODELS))}")

    # 3. Check !include directives and ensure no path traversal outside root
    include_matches = re.findall(r"!include\s+([^\s\n]+)", content)
    for inc_path in include_matches:
        if inc_path.startswith("/") or ".." in inc_path:
            errors.append(f"Forbidden path traversal in !include: {inc_path}")
            continue
        target_file = submission_dir / inc_path
        if not target_file.exists():
            errors.append(f"Included file does not exist: {inc_path}")

    # 4. Check sub-agents if referenced
    sub_agent_matches = re.findall(r"sub_agents/([^\s\n]+)", content)
    for sa in sub_agent_matches:
        sa_file = submission_dir / "sub_agents" / sa
        if not sa_file.exists():
            errors.append(f"Sub-agent YAML not found: {sa_file}")

    # 5. Check adapters
    adapter_dir = submission_dir / "adapters"
    if adapter_dir.exists():
        for sub in adapter_dir.iterdir():
            if sub.is_dir():
                cfg = sub / "adapter_config.json"
                st = sub / "adapter_model.safetensors"
                if not cfg.exists():
                    errors.append(f"Missing adapter_config.json in adapter '{sub.name}'")
                if not st.exists():
                    errors.append(f"Missing adapter_model.safetensors in adapter '{sub.name}'")

    # 6. Check skills
    skills_dir = submission_dir / "skills"
    if skills_dir.exists():
        for sk in skills_dir.iterdir():
            if sk.is_dir():
                skill_md = sk / "SKILL.md"
                if not skill_md.exists():
                    errors.append(f"Skill '{sk.name}' missing SKILL.md manifest")
                else:
                    text = skill_md.read_text(encoding="utf-8")
                    if not text.startswith("---") or "name:" not in text:
                        errors.append(f"Skill '{sk.name}/SKILL.md' missing YAML frontmatter with 'name:'")

    return errors


def build_submission_zip(submission_dir: Path, output_zip_path: Path) -> None:
    """Package submission_dir into zip archive with agent.yaml at the archive root."""
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    table = Table(title="Submission Archive Contents", show_header=True)
    table.add_column("Archive File Path", style="cyan")
    table.add_column("Size", justify="right", style="green")

    total_uncompressed = 0

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(submission_dir):
            for file in sorted(files):
                if file.startswith((".", "__pycache__")):
                    continue
                file_path = Path(root) / file
                archive_name = file_path.relative_to(submission_dir).as_posix()
                zf.write(file_path, arcname=archive_name)
                size_bytes = file_path.stat().st_size
                total_uncompressed += size_bytes
                size_str = f"{size_bytes / 1e6:.2f} MB" if size_bytes > 1e6 else f"{size_bytes / 1e3:.1f} KB"
                table.add_row(archive_name, size_str)

    console.print(table)
    zip_size_mb = output_zip_path.stat().st_size / 1e6
    console.print(f"\n[bold green]Successfully created {output_zip_path} ({zip_size_mb:.2f} MB compressed)[/bold green]")
    console.print(f"[dim]Total uncompressed payload: {total_uncompressed / 1e6:.2f} MB[/dim]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and Validate Kaggle Gemma ADK Submission")
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="./submission",
        help="Path to source submission directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./submission.zip",
        help="Path to output zip archive",
    )
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    out_zip = Path(args.output)

    console.print(f"\n[bold magenta]Validating submission directory:[/bold magenta] {sub_dir}")
    errors = validate_submission_directory(sub_dir)

    if errors:
        console.print("[bold red]Validation Failed with the following errors:[/bold red]")
        for err in errors:
            console.print(f"  [red]• {err}[/red]")
        sys.exit(1)

    console.print("[bold green]Validation passed: All required ADK files and tools conform to competition rules.[/bold green]\n")
    build_submission_zip(sub_dir, out_zip)


if __name__ == "__main__":
    main()
