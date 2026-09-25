"""Unit tests for Parallel-Distill-Refine (PDR), Dual-Suite Regression Verifier, and Checkpoint Persistence."""

import os
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from src.data.sri_formatter import SRIBlock, SRIFormatter
from src.evaluation.parallel_distill_refine import (
    CandidateHypothesis,
    ParallelDistillRefine,
    PDRResult,
)
from src.peft.dr_lora import DRLoRALinear, DRLoRAManager
from src.rlvr.prm import RubricProcessRewardModel
from src.rlvr.verifier import ExecutionSandboxVerifier, VerificationResult


def test_dual_suite_verifier_regression_detection():
    """Verify that ExecutionSandboxVerifier detects regressions in PASS_TO_PASS tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir) / "repo"
        repo.mkdir()

        # Target file
        (repo / "math_lib.py").write_text("def subtract(a, b):\n    return a + b\n")

        # Patch fixing math_lib.py
        patch_blocks = [
            SRIBlock(
                file_path="math_lib.py",
                search_content="    return a + b",
                replace_content="    return a - b",
            )
        ]

        verifier = ExecutionSandboxVerifier(timeout_seconds=5)

        # Case 1: FAIL_TO_PASS passes (exit 0), PASS_TO_PASS passes (exit 0)
        res1 = verifier.verify_dual_suite(
            base_dir=str(repo),
            patch_blocks=patch_blocks,
            fail_to_pass_command="python3 -c 'from math_lib import subtract; assert subtract(5, 2) == 3'",
            pass_to_pass_command="python3 -c 'assert 1 + 1 == 2'",
        )
        assert res1.passed is True
        assert res1.reward == 1.0
        assert res1.fail_to_pass_passed is True
        assert res1.pass_to_pass_passed is True
        assert res1.regression_detected is False

        # Case 2: FAIL_TO_PASS passes (exit 0), but PASS_TO_PASS fails (exit 1 -> regression!)
        res2 = verifier.verify_dual_suite(
            base_dir=str(repo),
            patch_blocks=patch_blocks,
            fail_to_pass_command="python3 -c 'from math_lib import subtract; assert subtract(5, 2) == 3'",
            pass_to_pass_command="python3 -c 'assert 1 + 1 == 999'",  # Simulating broken regression test
        )
        assert res2.passed is False
        assert res2.reward == 0.0
        assert res2.fail_to_pass_passed is True
        assert res2.pass_to_pass_passed is False
        assert res2.regression_detected is True
        assert "Regression detected" in (res2.error_message or "")


def test_pdr_failure_distillation():
    """Test that failure modes are accurately condensed into distilled feedback."""
    pdr = ParallelDistillRefine(k=3, max_refine_rounds=1)

    cands = [
        CandidateHypothesis(
            candidate_id=1,
            raw_output="I think the file has a bug in line 10.",
            sri_blocks=[],
            test_stderr="",
            error_summary="No SRI block found.",
        ),
        CandidateHypothesis(
            candidate_id=2,
            raw_output="""<<<<<<< SEARCH: calc.py
return a - b
=======
return a + b
>>>>>>> REPLACE""",
            sri_blocks=[SRIBlock("calc.py", "return a - b", "return a + b")],
            test_stderr="AssertionError: 5 != 10",
        ),
    ]

    distilled = pdr.distill_failure_analysis(cands)
    assert "Distilled Analysis of Prior Candidate Attempts" in distilled
    assert "Candidate #1" in distilled
    assert "Candidate #2" in distilled
    assert "AssertionError: 5 != 10" in distilled


def test_pdr_tournament_voting_selection():
    """Verify recursive tournament selection prioritizes verified, passing patches."""
    pdr = ParallelDistillRefine(k=2)

    cand_bad = CandidateHypothesis(
        candidate_id=1,
        raw_output="plain explanation",
        sri_blocks=[],
        prm_score=0.2,
        passed_tests=False,
        composite_score=-5.0,
    )
    cand_good = CandidateHypothesis(
        candidate_id=2,
        raw_output="""<<<<<<< SEARCH: calc.py
return a - b
=======
return a + b
>>>>>>> REPLACE""",
        sri_blocks=[SRIBlock("calc.py", "return a - b", "return a + b")],
        prm_score=1.0,
        passed_tests=True,
        composite_score=111.0,
    )

    winner = pdr.recursive_tournament_vote([cand_bad, cand_good])
    assert winner.candidate_id == 2
    assert winner.passed_tests is True


def test_dr_lora_save_and_load_checkpoint():
    """Verify that DR-LoRA checkpointing saves and restores adapter parameters and active ranks."""
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(32, 32)
            self.linear2 = nn.Linear(32, 32)

    model1 = DummyTransformer()
    manager1 = DRLoRAManager(
        model=model1,
        target_modules=["linear1", "linear2"],
        max_rank=16,
        min_rank=2,
        initial_rank=4,
    )

    # Change active rank of linear1
    manager1.adapted_layers["linear1"].set_active_rank(12)
    assert manager1.adapted_layers["linear1"].active_rank == 12

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_adapter.pt")
        manager1.save_adapters(ckpt_path)
        assert os.path.exists(ckpt_path)

        # Restore into a fresh model
        model2 = DummyTransformer()
        manager2 = DRLoRAManager.from_checkpoint(model2, ckpt_path)

        assert "linear1" in manager2.adapted_layers
        assert manager2.adapted_layers["linear1"].active_rank == 12
        assert manager2.adapted_layers["linear2"].active_rank == 4
        # Verify weights match
        assert torch.allclose(
            manager1.adapted_layers["linear1"].lora_A,
            manager2.adapted_layers["linear1"].lora_A,
        )
