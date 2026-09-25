"""SWE-bench Evaluation Runner and Orchestrator.

Manages prompt generation, model inference, multi-turn agent exploration,
patch extraction, dual-suite test execution (FAIL_TO_PASS + PASS_TO_PASS),
and Test-Time Scaling (TTS / Parallel-Distill-Refine).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data.sri_formatter import SRIBlock, SRIFormatter
from src.evaluation.metrics import InstanceEvaluationResult, MetricsCalculator, BenchmarkSummary
from src.evaluation.patch_applicator import PatchApplicator
from src.evaluation.swe_bench_loader import SWEBenchInstance


class SWEBenchRunner:
    """Evaluates agent models on SWE-bench instances."""

    SYSTEM_PROMPT = (
        "You are an expert autonomous software engineer. You are given an issue from a repository.\n"
        "Analyze the bug, isolate the fault location, and output a fix using the Search-and-Replace Infilling (SRI) format:\n\n"
        "<<<<<<< SEARCH: relative/path/to/file.py\n"
        "<exact code to search for>\n"
        "=======\n"
        "<replacement code>\n"
        ">>>>>>> REPLACE\n\n"
        "Ensure your search block is exact and unique. Do not include line numbers or unified diff markers."
    )

    def __init__(
        self,
        model_generate_fn: Optional[Callable[[str], str]] = None,
        timeout_seconds: int = 120,
        verifier: Optional[Any] = None,
    ):
        """Args:
            model_generate_fn: Function that accepts a prompt string and returns the generated text.
            timeout_seconds: Maximum execution time per test suite run.
            verifier: Optional ExecutionSandboxVerifier (e.g. DockerSandboxVerifier) for running tests.
        """
        self.model_generate_fn = model_generate_fn
        self.timeout_seconds = timeout_seconds
        self.verifier = verifier

    def format_instance_prompt(self, instance: SWEBenchInstance) -> str:
        """Construct the prompt text for an instance."""
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Repository: {instance.repo}\n"
            f"Base Commit: {instance.base_commit}\n"
            f"Problem Statement:\n{instance.problem_statement}\n\n"
            f"Please provide your fix in SRI format:"
        )

    def evaluate_instance(
        self,
        instance: SWEBenchInstance,
        repo_dir: str,
        test_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
        pass_to_pass_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
        generated_patch: Optional[str] = None,
    ) -> InstanceEvaluationResult:
        """Run evaluation for a single instance against a local repo clone/sandbox.

        Verifies both FAIL_TO_PASS (must pass) and PASS_TO_PASS (no regression) suites.
        """
        start_time = time.time()

        # 1. Obtain model generation if not directly provided
        if generated_patch is None:
            if self.model_generate_fn is None:
                raise ValueError("Either generated_patch or model_generate_fn must be provided.")
            prompt = self.format_instance_prompt(instance)
            model_output = self.model_generate_fn(prompt)
        else:
            model_output = generated_patch

        # 2. Apply patch to repository
        patch_res = PatchApplicator.extract_and_apply(model_output, repo_dir)
        if not patch_res.success:
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=False,
                format_type=patch_res.format_type,
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                execution_time_seconds=time.time() - start_time,
                error_message=f"Patch application failed: {patch_res.error_message}",
            )

        # 3. Determine test command(s)
        if test_command_fn is not None:
            f2p_cmd = test_command_fn(instance)
        else:
            test_targets = " ".join(shlex.quote(t) for t in instance.fail_to_pass) if instance.fail_to_pass else ""
            f2p_cmd = f"pytest {test_targets}" if test_targets else "pytest"

        p2p_cmd = None
        if pass_to_pass_command_fn is not None:
            p2p_cmd = pass_to_pass_command_fn(instance)
        elif instance.pass_to_pass:
            p2p_targets = " ".join(shlex.quote(t) for t in instance.pass_to_pass[:10])
            p2p_cmd = f"pytest {p2p_targets}"

        # If an external verifier (e.g. DockerSandboxVerifier) is configured, delegate
        if self.verifier is not None:
            v_res = self.verifier.verify_dual_suite(
                base_dir=repo_dir,
                patch_blocks=patch_res.sri_blocks,
                fail_to_pass_command=f2p_cmd,
                pass_to_pass_command=p2p_cmd,
                repo_name=instance.repo,
            )
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=v_res.passed,
                patch_applied=v_res.patch_applied,
                format_type=patch_res.format_type,
                fail_to_pass_passed=v_res.fail_to_pass_passed,
                pass_to_pass_passed=v_res.pass_to_pass_passed,
                execution_time_seconds=time.time() - start_time,
                error_message=v_res.error_message,
            )

        # 4. Execute FAIL_TO_PASS test command
        try:
            f2p_proc = subprocess.run(
                f2p_cmd,
                shell=True,
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
            f2p_passed = (f2p_proc.returncode == 0)
        except subprocess.TimeoutExpired:
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=True,
                format_type=patch_res.format_type,
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                execution_time_seconds=time.time() - start_time,
                error_message=f"FAIL_TO_PASS timed out after {self.timeout_seconds}s.",
            )
        except Exception as e:
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=True,
                format_type=patch_res.format_type,
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                execution_time_seconds=time.time() - start_time,
                error_message=f"FAIL_TO_PASS execution error: {e}",
            )

        # If fail-to-pass failed, the patch did not fix the bug
        if not f2p_passed:
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=True,
                format_type=patch_res.format_type,
                fail_to_pass_passed=False,
                pass_to_pass_passed=True,
                execution_time_seconds=time.time() - start_time,
                error_message=f"FAIL_TO_PASS tests failed with exit code {f2p_proc.returncode}",
            )

        # 5. Check PASS_TO_PASS regression tests
        p2p_passed = True
        p2p_err = None

        if pass_to_pass_command_fn is not None:
            p2p_cmd = pass_to_pass_command_fn(instance)
        elif instance.pass_to_pass:
            p2p_cmd = f"pytest {' '.join(instance.pass_to_pass)}"
        else:
            p2p_cmd = None

        if p2p_cmd:
            try:
                p2p_proc = subprocess.run(
                    p2p_cmd,
                    shell=True,
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                p2p_passed = (p2p_proc.returncode == 0)
                if not p2p_passed:
                    p2p_err = f"PASS_TO_PASS regression: exit code {p2p_proc.returncode}"
            except subprocess.TimeoutExpired:
                p2p_passed = False
                p2p_err = "PASS_TO_PASS regression test timed out."
            except Exception as e:
                p2p_passed = False
                p2p_err = f"PASS_TO_PASS execution error: {e}"

        resolved = f2p_passed and p2p_passed
        return InstanceEvaluationResult(
            instance_id=instance.instance_id,
            resolved=resolved,
            patch_applied=True,
            format_type=patch_res.format_type,
            fail_to_pass_passed=f2p_passed,
            pass_to_pass_passed=p2p_passed,
            execution_time_seconds=time.time() - start_time,
            error_message=p2p_err if not resolved else None,
        )

    def evaluate_instance_agentic(
        self,
        instance: SWEBenchInstance,
        repo_dir: str,
        max_turns: int = 6,
        test_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
        pass_to_pass_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
        tokenizer: Optional[Any] = None,
    ) -> InstanceEvaluationResult:
        """Run interactive multi-turn agent exploration and bug fixing for an instance."""
        if self.model_generate_fn is None:
            raise ValueError("model_generate_fn must be provided for agentic evaluation.")

        from src.agent.agent_loop import MultiTurnAgent

        agent = MultiTurnAgent(
            generate_fn=self.model_generate_fn,
            tokenizer=tokenizer,
            max_turns=max_turns,
        )
        session = agent.run_session(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            repo_dir=repo_dir,
        )

        patch_text = session.final_patch_raw or ""
        if not patch_text:
            return InstanceEvaluationResult(
                instance_id=instance.instance_id,
                resolved=False,
                patch_applied=False,
                format_type="none",
                fail_to_pass_passed=False,
                pass_to_pass_passed=False,
                error_message=f"Agent completed {session.turns} turns without submitting a patch.",
            )

        return self.evaluate_instance(
            instance=instance,
            repo_dir=repo_dir,
            test_command_fn=test_command_fn,
            pass_to_pass_command_fn=pass_to_pass_command_fn,
            generated_patch=patch_text,
        )

    def evaluate_instance_pdr(
        self,
        instance: SWEBenchInstance,
        repo_dir: str,
        k: int = 4,
        sample_generate_fn: Optional[Callable[[str, int], List[str]]] = None,
        test_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
    ) -> Tuple[InstanceEvaluationResult, bool, str]:
        """Evaluate an instance using Parallel-Distill-Refine (PDR) Test-Time Scaling."""
        from src.evaluation.parallel_distill_refine import ParallelDistillRefine

        pdr = ParallelDistillRefine(k=k, max_refine_rounds=1)
        prompt = self.format_instance_prompt(instance)

        def batch_gen(p: str, count: int) -> List[str]:
            if sample_generate_fn is not None:
                return sample_generate_fn(p, count)
            elif self.model_generate_fn is not None:
                return [self.model_generate_fn(p) for _ in range(count)]
            raise ValueError("No candidate generation function provided.")

        def verifier_cb(blocks: List[SRIBlock]) -> Tuple[bool, str, str]:
            from src.rlvr.verifier import ExecutionSandboxVerifier

            sb = ExecutionSandboxVerifier(timeout_seconds=self.timeout_seconds)
            t_cmd = test_command_fn(instance) if test_command_fn else "pytest"
            res = sb.verify_patch(repo_dir, blocks, t_cmd)
            return res.passed, res.stdout, res.stderr

        pdr_result = pdr.run_pdr(
            initial_prompt=prompt,
            generate_fn=batch_gen,
            verify_fn=verifier_cb,
        )

        winning_patch = pdr_result.winning_candidate.raw_output
        eval_res = self.evaluate_instance(
            instance=instance,
            repo_dir=repo_dir,
            test_command_fn=test_command_fn,
            generated_patch=winning_patch,
        )

        return eval_res, pdr_result.pass_at_k, pdr_result.distilled_feedback

    def evaluate_instance_tts(
        self,
        instance: SWEBenchInstance,
        repo_dir: str,
        k: int = 4,
        sample_generate_fn: Optional[Callable[[str, int], List[str]]] = None,
        test_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
    ) -> Tuple[InstanceEvaluationResult, bool]:
        """Evaluate an instance using standard Test-Time Scaling across K candidate rollouts."""
        eval_res, pass_at_k, _ = self.evaluate_instance_pdr(
            instance=instance,
            repo_dir=repo_dir,
            k=k,
            sample_generate_fn=sample_generate_fn,
            test_command_fn=test_command_fn,
        )
        return eval_res, pass_at_k

    def run_benchmark(
        self,
        instances: List[SWEBenchInstance],
        repo_dirs_by_id: Dict[str, str],
        test_command_fn: Optional[Callable[[SWEBenchInstance], str]] = None,
        agentic: bool = False,
    ) -> BenchmarkSummary:
        """Run batch evaluation over a list of SWE-bench instances."""
        results: List[InstanceEvaluationResult] = []

        for inst in instances:
            repo_dir = repo_dirs_by_id.get(inst.instance_id)
            if not repo_dir:
                results.append(
                    InstanceEvaluationResult(
                        instance_id=inst.instance_id,
                        resolved=False,
                        patch_applied=False,
                        format_type="none",
                        fail_to_pass_passed=False,
                        pass_to_pass_passed=False,
                        error_message=f"No local repo directory mapped for instance {inst.instance_id}",
                    )
                )
                continue

            if agentic:
                res = self.evaluate_instance_agentic(
                    instance=inst,
                    repo_dir=repo_dir,
                    test_command_fn=test_command_fn,
                )
            else:
                res = self.evaluate_instance(
                    instance=inst,
                    repo_dir=repo_dir,
                    test_command_fn=test_command_fn,
                )
            results.append(res)

        return MetricsCalculator.compute_summary(results)
