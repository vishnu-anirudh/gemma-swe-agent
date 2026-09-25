"""Patch extractor and applicator for SWE-bench evaluation.

Supports Search-and-Replace Infilling (SRI) blocks and unified diffs,
handling formatting validation and application to repository directories.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.sri_formatter import SRIBlock, SRIFormatter


@dataclass
class PatchApplicationResult:
    """Result of attempting to apply a patch to a repository."""

    success: bool
    format_type: str  # 'sri' or 'diff'
    applied_files: List[str]
    error_message: Optional[str] = None
    unified_diff: Optional[str] = None


class PatchApplicator:
    """Extracts patches from model responses and applies them to a repository path."""

    @classmethod
    def extract_and_apply(
        cls,
        raw_model_response: str,
        repo_dir: str,
    ) -> PatchApplicationResult:
        """Analyze model response, determine format, and apply changes to repo_dir."""
        repo_path = Path(repo_dir)

        # 1. First attempt: Check for SRI blocks (preferred format)
        sri_blocks = SRIFormatter.parse(raw_model_response)
        if len(sri_blocks) > 0:
            return cls._apply_sri_blocks(sri_blocks, repo_path)

        # 2. Second attempt: Check for unified diff
        if "diff --git" in raw_model_response or "--- a/" in raw_model_response:
            return cls._apply_unified_diff(raw_model_response, repo_path)

        return PatchApplicationResult(
            success=False,
            format_type="unknown",
            applied_files=[],
            error_message="Could not parse any SRI blocks or unified diff from model response.",
        )

    @classmethod
    def _apply_sri_blocks(
        cls,
        blocks: List[SRIBlock],
        repo_path: Path,
    ) -> PatchApplicationResult:
        applied_files: List[str] = []
        original_contents: Dict[str, str] = {}
        modified_contents: Dict[str, str] = {}

        for idx, block in enumerate(blocks):
            target_file = repo_path / block.file_path
            if not target_file.exists():
                return PatchApplicationResult(
                    success=False,
                    format_type="sri",
                    applied_files=applied_files,
                    error_message=f"Block {idx+1}: File '{block.file_path}' does not exist in repo.",
                )

            try:
                content = target_file.read_text(encoding="utf-8", errors="replace")
                if block.file_path not in original_contents:
                    original_contents[block.file_path] = content

                new_content, success, msg = SRIFormatter.apply_to_content(
                    content, block.search_content, block.replace_content
                )

                if not success:
                    return PatchApplicationResult(
                        success=False,
                        format_type="sri",
                        applied_files=applied_files,
                        error_message=f"Block {idx+1} for '{block.file_path}' failed: {msg}",
                    )

                target_file.write_text(new_content, encoding="utf-8")
                modified_contents[block.file_path] = new_content
                if block.file_path not in applied_files:
                    applied_files.append(block.file_path)

            except Exception as e:
                return PatchApplicationResult(
                    success=False,
                    format_type="sri",
                    applied_files=applied_files,
                    error_message=f"Error reading/writing '{block.file_path}': {e}",
                )

        diff = SRIFormatter.to_unified_diff(original_contents, modified_contents)
        return PatchApplicationResult(
            success=True,
            format_type="sri",
            applied_files=applied_files,
            unified_diff=diff,
        )

    @classmethod
    def _apply_unified_diff(
        cls,
        diff_text: str,
        repo_path: Path,
    ) -> PatchApplicationResult:
        # Extract diff lines
        lines = diff_text.splitlines()
        clean_diff_lines = []
        capturing = False

        for line in lines:
            if line.startswith("diff --git") or line.startswith("--- a/"):
                capturing = True
            if capturing:
                clean_diff_lines.append(line)

        clean_diff = "\n".join(clean_diff_lines) + "\n"
        if not clean_diff.strip():
            return PatchApplicationResult(
                success=False,
                format_type="diff",
                applied_files=[],
                error_message="Found diff markers but could not extract clean diff content.",
            )

        # Apply using git apply
        cmd = ["git", "apply", "--verbose", "--whitespace=nowarn", "-"]
        proc = subprocess.run(
            cmd,
            input=clean_diff,
            cwd=str(repo_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode == 0:
            return PatchApplicationResult(
                success=True,
                format_type="diff",
                applied_files=[],
                unified_diff=clean_diff,
            )

        return PatchApplicationResult(
            success=False,
            format_type="diff",
            applied_files=[],
            error_message=f"git apply failed: {proc.stderr}",
            unified_diff=clean_diff,
        )
