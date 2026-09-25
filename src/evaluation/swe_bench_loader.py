"""SWE-bench dataset loader and instance parser.

Supports loading SWE-bench Lite and SWE-bench Verified instances from
Hugging Face datasets or local cached JSONL files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SWEBenchInstance:
    """Represents a single issue instance from SWE-bench."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    hints_text: Optional[str] = None
    created_at: Optional[str] = None
    version: Optional[str] = None
    environment_setup_commit: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SWEBenchInstance:
        """Parse raw dictionary (from Hugging Face datasets or JSON) into typed instance."""
        f2p = data.get("FAIL_TO_PASS", [])
        if isinstance(f2p, str):
            try:
                f2p = json.loads(f2p)
            except Exception:
                f2p = [f2p]

        p2p = data.get("PASS_TO_PASS", [])
        if isinstance(p2p, str):
            try:
                p2p = json.loads(p2p)
            except Exception:
                p2p = [p2p]

        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            problem_statement=data.get("problem_statement", ""),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            hints_text=data.get("hints_text"),
            created_at=data.get("created_at"),
            version=data.get("version"),
            environment_setup_commit=data.get("environment_setup_commit"),
        )


class SWEBenchLoader:
    """Loads benchmark instances from Hugging Face or local files."""

    DATASET_LITE = "princeton-nlp/SWE-bench_Lite"
    DATASET_VERIFIED = "princeton-nlp/SWE-bench_Verified"

    @classmethod
    def load_from_huggingface(
        cls,
        dataset_name: str = DATASET_LITE,
        split: str = "test",
        limit: Optional[int] = None,
    ) -> List[SWEBenchInstance]:
        """Load instances directly from Hugging Face datasets."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        instances: List[SWEBenchInstance] = []

        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            instances.append(SWEBenchInstance.from_dict(row))

        return instances

    @classmethod
    def load_from_file(cls, file_path: str, limit: Optional[int] = None) -> List[SWEBenchInstance]:
        """Load instances from a local JSON or JSONL file."""
        path = Path(file_path)
        instances: List[SWEBenchInstance] = []

        if not path.exists():
            raise FileNotFoundError(f"SWE-bench file not found at: {file_path}")

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if limit is not None and idx >= limit:
                        break
                    line = line.strip()
                    if line:
                        instances.append(SWEBenchInstance.from_dict(json.loads(line)))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, row in enumerate(data):
                        if limit is not None and idx >= limit:
                            break
                        instances.append(SWEBenchInstance.from_dict(row))
                elif isinstance(data, dict):
                    instances.append(SWEBenchInstance.from_dict(data))

        return instances
