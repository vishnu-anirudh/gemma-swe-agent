"""Unit tests for SWE-bench loader, applicator, and evaluation runner."""

import tempfile
from pathlib import Path
from src.evaluation.metrics import InstanceEvaluationResult, MetricsCalculator
from src.evaluation.patch_applicator import PatchApplicator
from src.evaluation.swe_bench_loader import SWEBenchInstance
from src.evaluation.swe_bench_runner import SWEBenchRunner


def test_swe_bench_instance_from_dict():
    raw_dict = {
        "instance_id": "psf__requests-2148",
        "repo": "psf/requests",
        "base_commit": "a1b2c3d",
        "problem_statement": "Socket error in connection pool",
        "FAIL_TO_PASS": '["tests/test_requests.py::test_socket_error"]',
        "PASS_TO_PASS": '["tests/test_requests.py::test_get"]',
    }
    inst = SWEBenchInstance.from_dict(raw_dict)
    assert inst.instance_id == "psf__requests-2148"
    assert inst.repo == "psf/requests"
    assert len(inst.fail_to_pass) == 1
    assert inst.fail_to_pass[0] == "tests/test_requests.py::test_socket_error"


def test_patch_applicator_with_sri_blocks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir)
        test_file = repo_path / "module.py"
        test_file.write_text("x = 10\ny = 20\n", encoding="utf-8")

        sri_response = """
Here is the solution:
<<<<<<< SEARCH: module.py
x = 10
=======
x = 99
>>>>>>> REPLACE
"""
        res = PatchApplicator.extract_and_apply(sri_response, str(repo_path))
        assert res.success is True
        assert res.format_type == "sri"
        assert test_file.read_text(encoding="utf-8") == "x = 99\ny = 20\n"
        assert res.unified_diff is not None
        assert "-x = 10" in res.unified_diff
        assert "+x = 99" in res.unified_diff


def test_metrics_calculator_pass_at_1_and_k():
    results = [
        InstanceEvaluationResult(
            instance_id="inst_1",
            resolved=True,
            patch_applied=True,
            format_type="sri",
            fail_to_pass_passed=True,
            pass_to_pass_passed=True,
        ),
        InstanceEvaluationResult(
            instance_id="inst_2",
            resolved=False,
            patch_applied=True,
            format_type="sri",
            fail_to_pass_passed=False,
            pass_to_pass_passed=True,
        ),
        InstanceEvaluationResult(
            instance_id="inst_3",
            resolved=False,
            patch_applied=False,
            format_type="unknown",
            fail_to_pass_passed=False,
            pass_to_pass_passed=False,
        ),
    ]

    summary = MetricsCalculator.compute_summary(results)
    assert summary.total_instances == 3
    assert summary.resolved_instances == 1
    assert summary.unresolved_instances == 2
    assert abs(summary.pass_at_1 - (1 / 3)) < 1e-4
    assert abs(summary.patch_application_rate - (2 / 3)) < 1e-4

    # Test Pass@K calculation
    p_at_1 = MetricsCalculator.compute_pass_at_k(n=10, c=2, k=1)
    p_at_5 = MetricsCalculator.compute_pass_at_k(n=10, c=2, k=5)
    assert p_at_1 < p_at_5
