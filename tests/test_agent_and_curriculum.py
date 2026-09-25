"""Unit tests for MultiTurnAgent, RepoTools, Curriculum Dataset, and TTS."""

import tempfile
from pathlib import Path
from src.agent.agent_loop import MultiTurnAgent
from src.agent.tools import RepoTools
from src.data.curriculum import CurriculumTrajectoryDataset
from src.data.masking import StepLevelErrorMasker
from src.evaluation.swe_bench_loader import SWEBenchInstance
from src.evaluation.swe_bench_runner import SWEBenchRunner


def test_repo_tools_view_and_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        test_file = repo / "sample.py"
        test_file.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

        # Test view_file
        view_out = RepoTools.view_file(str(repo), "sample.py", start_line=2, end_line=3)
        assert "line2" in view_out
        assert "line3" in view_out
        assert "line1" not in view_out

        # Test list_dir
        list_out = RepoTools.list_dir(str(repo))
        assert "sample.py" in list_out


def test_multi_turn_agent_tool_calling_and_patch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        (repo / "math_mod.py").write_text("def subtract(a, b):\n    return a + b\n", encoding="utf-8")

        turn_responses = [
            '<tool_call name="view_file" path="math_mod.py" start_line="1" end_line="5" />',
            '<submit_patch>\n<<<<<<< SEARCH: math_mod.py\n    return a + b\n=======\n    return a - b\n>>>>>>> REPLACE\n</submit_patch>',
        ]
        turn_counter = 0

        def mock_generate(prompt: str) -> str:
            nonlocal turn_counter
            resp = turn_responses[min(turn_counter, len(turn_responses) - 1)]
            turn_counter += 1
            return resp

        agent = MultiTurnAgent(generate_fn=mock_generate, max_turns=4)
        session = agent.run_session(
            instance_id="toy_test_1",
            problem_statement="Subtract function has wrong operator.",
            repo_dir=str(repo),
        )

        assert session.turns == 2
        assert session.tool_calls_made == 1
        assert len(session.parsed_sri_blocks) == 1
        assert session.parsed_sri_blocks[0].file_path == "math_mod.py"


def test_curriculum_trajectory_dataset():
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 1000 for w in text.split()]

    masker = StepLevelErrorMasker(tokenizer=DummyTokenizer())
    dataset = CurriculumTrajectoryDataset.create_synthetic_curriculum(masker)

    assert len(dataset) == 3
    # Check ordering: easy, medium, hard
    assert dataset.trajectories[0].difficulty == "easy"
    assert dataset.trajectories[1].difficulty == "medium"
    assert dataset.trajectories[2].difficulty == "hard"

    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == item["labels"].shape


def test_expanded_curriculum_dataset():
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 1000 for w in text.split()]

    masker = StepLevelErrorMasker(tokenizer=DummyTokenizer())
    dataset = CurriculumTrajectoryDataset.generate_expanded_curriculum(masker)

    assert len(dataset) == 30
    easy_count = sum(1 for t in dataset.trajectories if t.difficulty == "easy")
    med_count = sum(1 for t in dataset.trajectories if t.difficulty == "medium")
    hard_count = sum(1 for t in dataset.trajectories if t.difficulty == "hard")

    assert easy_count == 10
    assert med_count == 10
    assert hard_count == 10


def test_curriculum_from_jsonl():
    import json
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 1000 for w in text.split()]

    masker = StepLevelErrorMasker(tokenizer=DummyTokenizer())
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        data = {
            "instance_id": "test_jsonl_1",
            "difficulty": "medium",
            "system_prompt": "Fix bug",
            "steps": [
                {"action_text": "search_code query='test'", "observation_text": "found", "is_error": False}
            ],
            "final_patch": "<<<<<<< SEARCH: a.py\n1\n=======\n2\n>>>>>>> REPLACE",
        }
        f.write(json.dumps(data) + "\n")
        f_path = f.name

    try:
        ds = CurriculumTrajectoryDataset.from_jsonl(f_path, masker)
        assert len(ds) == 1
        assert ds.trajectories[0].instance_id == "test_jsonl_1"
        assert ds.trajectories[0].difficulty == "medium"
    finally:
        Path(f_path).unlink(missing_ok=True)
