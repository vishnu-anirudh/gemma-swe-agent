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


def test_repo_manager_sandbox_lifecycle():
    import subprocess
    from src.evaluation.repo_manager import RepoManager

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock upstream repo
        origin_dir = Path(tmp_dir) / "mock_origin"
        origin_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(origin_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(origin_dir), check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(origin_dir), check=True)

        (origin_dir / "app.py").write_text("v = 1\n")
        subprocess.run(["git", "add", "app.py"], cwd=str(origin_dir), check=True)
        subprocess.run(["git", "commit", "-m", "commit 1"], cwd=str(origin_dir), check=True)
        rev1 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(origin_dir), text=True).strip()

        (origin_dir / "app.py").write_text("v = 2\n")
        subprocess.run(["git", "commit", "-am", "commit 2"], cwd=str(origin_dir), check=True)

        # Create sandbox checked out at rev1
        cache_dir = Path(tmp_dir) / "cache"
        sandbox = RepoManager.create_instance_sandbox(
            repo_name=str(origin_dir),
            base_commit=rev1,
            target_dir=Path(tmp_dir) / "instance_sb",
            cache_dir=cache_dir,
        )
        assert sandbox.exists()
        assert (sandbox / "app.py").read_text() == "v = 1\n"

        # Cleanup
        RepoManager.remove_sandbox(sandbox)
        assert not sandbox.exists()


def test_swe_bench_runner_build_test_command():
    # Django instance test formatting
    django_inst = SWEBenchInstance(
        instance_id="django__django-10914",
        repo="django/django",
        base_commit="e7fd69",
        problem_statement="Test permissions",
        fail_to_pass=["test_override_file_upload_permissions (test_utils.tests.OverrideSettingsTests)"],
        pass_to_pass=["test_other (test_utils.tests.OtherTests)"],
    )
    cmd = SWEBenchRunner.build_test_command(django_inst)
    assert "tests/runtests.py" in cmd
    assert "test_utils.tests.OverrideSettingsTests.test_override_file_upload_permissions" in cmd

    # Generic instance test formatting
    generic_inst = SWEBenchInstance(
        instance_id="psf__requests-100",
        repo="psf/requests",
        base_commit="abc",
        problem_statement="Test requests",
        fail_to_pass=["tests/test_requests.py::test_get"],
    )
    generic_cmd = SWEBenchRunner.build_test_command(generic_inst)
    assert "pytest" in generic_cmd
    assert "tests/test_requests.py::test_get" in generic_cmd


def test_repo_manager_with_test_patch():
    import subprocess
    from src.evaluation.repo_manager import RepoManager

    with tempfile.TemporaryDirectory() as tmp_dir:
        origin_dir = Path(tmp_dir) / "mock_origin"
        origin_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(origin_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(origin_dir), check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(origin_dir), check=True)

        (origin_dir / "test_suite.py").write_text("def test_old(): pass\n")
        subprocess.run(["git", "add", "."], cwd=str(origin_dir), check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(origin_dir), check=True)
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(origin_dir), text=True).strip()

        # Create a test patch
        test_patch = """diff --git a/test_suite.py b/test_suite.py
--- a/test_suite.py
+++ b/test_suite.py
@@ -1 +1,2 @@
 def test_old(): pass
+def test_new_case(): assert True
"""
        sandbox = RepoManager.create_instance_sandbox(
            repo_name=str(origin_dir),
            base_commit=rev,
            cache_dir=Path(tmp_dir) / "cache",
            test_patch=test_patch,
        )
        try:
            assert "test_new_case" in (sandbox / "test_suite.py").read_text()
        finally:
            RepoManager.remove_sandbox(sandbox)


