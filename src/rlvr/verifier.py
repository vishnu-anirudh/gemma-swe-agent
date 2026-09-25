"""Verifiable execution verifier for RLVR rewards.

Executes code patches against local test suites in a secured,
time-limited subprocess sandbox to produce objective binary rewards,
enforcing both FAIL_TO_PASS resolution and PASS_TO_PASS regression protection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.sri_formatter import SRIBlock, SRIFormatter


@dataclass
class VerificationResult:
    """Outcome of running a patch against tests."""

    passed: bool
    reward: float  # 1.0 for pass, 0.0 for fail
    exit_code: int
    stdout: str
    stderr: str
    patch_applied: bool
    fail_to_pass_passed: bool = False
    pass_to_pass_passed: bool = True
    regression_detected: bool = False
    error_message: Optional[str] = None


class ExecutionSandboxVerifier:
    """Verifies candidate patches by executing test suites in ephemeral directories."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def verify_patch(
        self,
        base_dir: str,
        patch_blocks: List[SRIBlock],
        test_command: str,
    ) -> VerificationResult:
        """Apply patch to a temporary copy of base_dir and execute the test command."""
        return self.verify_dual_suite(
            base_dir=base_dir,
            patch_blocks=patch_blocks,
            fail_to_pass_command=test_command,
            pass_to_pass_command=None,
        )

    def verify_dual_suite(
        self,
        base_dir: str,
        patch_blocks: List[SRIBlock],
        fail_to_pass_command: str,
        pass_to_pass_command: Optional[str] = None,
    ) -> VerificationResult:
        """Apply patch and verify against both FAIL_TO_PASS and PASS_TO_PASS suites.

        A patch is only considered resolved (reward=1.0) if:
        1. All patch blocks apply cleanly.
        2. All FAIL_TO_PASS tests pass (exit code 0).
        3. All PASS_TO_PASS regression tests continue to pass (no regressions).
        """
        # Create an ephemeral sandbox directory to avoid modifying the base repo
        with tempfile.TemporaryDirectory(prefix="gemma_sandbox_") as tmp_dir:
            # Copy base repository
            dest_dir = Path(tmp_dir) / "repo"
            try:
                shutil.copytree(
                    base_dir,
                    dest_dir,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"),
                )
            except Exception as e:
                return VerificationResult(
                    passed=False,
                    reward=0.0,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    patch_applied=False,
                    fail_to_pass_passed=False,
                    pass_to_pass_passed=False,
                    regression_detected=False,
                    error_message=f"Failed to copy base repository: {e}",
                )

            # Apply SRI patch blocks
            applied_any = False
            for block in patch_blocks:
                target_file = dest_dir / block.file_path
                if not target_file.exists():
                    return VerificationResult(
                        passed=False,
                        reward=0.0,
                        exit_code=-1,
                        stdout="",
                        stderr=f"Target file {block.file_path} not found.",
                        patch_applied=False,
                        fail_to_pass_passed=False,
                        pass_to_pass_passed=False,
                        regression_detected=False,
                        error_message=f"File {block.file_path} does not exist in repo.",
                    )

                content = target_file.read_text(encoding="utf-8", errors="replace")
                new_content, success, msg = SRIFormatter.apply_to_content(
                    content, block.search_content, block.replace_content
                )
                if not success:
                    return VerificationResult(
                        passed=False,
                        reward=0.0,
                        exit_code=-1,
                        stdout="",
                        stderr=msg,
                        patch_applied=False,
                        fail_to_pass_passed=False,
                        pass_to_pass_passed=False,
                        regression_detected=False,
                        error_message=f"Patch application failed: {msg}",
                    )
                target_file.write_text(new_content, encoding="utf-8")
                applied_any = True

            # Step 1: Run FAIL_TO_PASS test suite
            try:
                f2p_proc = subprocess.run(
                    fail_to_pass_command,
                    shell=True,
                    cwd=str(dest_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                f2p_passed = (f2p_proc.returncode == 0)
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    passed=False,
                    reward=0.0,
                    exit_code=-1,
                    stdout="",
                    stderr=f"FAIL_TO_PASS test timed out after {self.timeout_seconds}s.",
                    patch_applied=applied_any,
                    fail_to_pass_passed=False,
                    pass_to_pass_passed=False,
                    regression_detected=False,
                    error_message="FAIL_TO_PASS test execution timed out.",
                )
            except Exception as e:
                return VerificationResult(
                    passed=False,
                    reward=0.0,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    patch_applied=applied_any,
                    fail_to_pass_passed=False,
                    pass_to_pass_passed=False,
                    regression_detected=False,
                    error_message=f"FAIL_TO_PASS execution error: {e}",
                )

            if not f2p_passed:
                return VerificationResult(
                    passed=False,
                    reward=0.0,
                    exit_code=f2p_proc.returncode,
                    stdout=f2p_proc.stdout,
                    stderr=f2p_proc.stderr,
                    patch_applied=applied_any,
                    fail_to_pass_passed=False,
                    pass_to_pass_passed=True,
                    regression_detected=False,
                    error_message=f"FAIL_TO_PASS test failed with exit code {f2p_proc.returncode}",
                )

            # Step 2: Run PASS_TO_PASS regression suite if provided
            p2p_passed = True
            combined_stdout = f2p_proc.stdout
            combined_stderr = f2p_proc.stderr
            regression = False

            if pass_to_pass_command:
                try:
                    p2p_proc = subprocess.run(
                        pass_to_pass_command,
                        shell=True,
                        cwd=str(dest_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                    p2p_passed = (p2p_proc.returncode == 0)
                    combined_stdout += f"\n--- PASS_TO_PASS ---\n{p2p_proc.stdout}"
                    combined_stderr += f"\n--- PASS_TO_PASS ---\n{p2p_proc.stderr}"
                    if not p2p_passed:
                        regression = True
                except subprocess.TimeoutExpired:
                    return VerificationResult(
                        passed=False,
                        reward=0.0,
                        exit_code=-1,
                        stdout=combined_stdout,
                        stderr="PASS_TO_PASS regression test timed out.",
                        patch_applied=applied_any,
                        fail_to_pass_passed=True,
                        pass_to_pass_passed=False,
                        regression_detected=True,
                        error_message="Regression test timed out.",
                    )
                except Exception as e:
                    return VerificationResult(
                        passed=False,
                        reward=0.0,
                        exit_code=-1,
                        stdout=combined_stdout,
                        stderr=str(e),
                        patch_applied=applied_any,
                        fail_to_pass_passed=True,
                        pass_to_pass_passed=False,
                        regression_detected=True,
                        error_message=f"PASS_TO_PASS execution error: {e}",
                    )

            overall_passed = f2p_passed and p2p_passed
            return VerificationResult(
                passed=overall_passed,
                reward=1.0 if overall_passed else 0.0,
                exit_code=0 if overall_passed else 1,
                stdout=combined_stdout,
                stderr=combined_stderr,
                patch_applied=applied_any,
                fail_to_pass_passed=f2p_passed,
                pass_to_pass_passed=p2p_passed,
                regression_detected=regression,
                error_message=None if overall_passed else ("Regression detected in PASS_TO_PASS test suite" if regression else "Test failed"),
            )
