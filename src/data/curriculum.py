"""Curriculum Learning Dataset Loader for Multi-Turn Trajectory SFT.

Categorizes software engineering trajectories by difficulty (interaction turns)
and applies Step-Level Error Masking for progressive curriculum training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
import torch
from torch.utils.data import Dataset

from src.data.masking import StepLevelErrorMasker, TrajectoryStep


@dataclass
class CuratedTrajectory:
    """Represents a complete multi-turn software engineering trajectory."""

    instance_id: str
    difficulty: str  # 'easy', 'medium', 'hard'
    system_prompt: str
    steps: List[TrajectoryStep]
    final_patch: str


class CurriculumTrajectoryDataset(Dataset):
    """Dataset providing trajectories organized by difficulty curriculum."""

    def __init__(
        self,
        trajectories: List[CuratedTrajectory],
        masker: StepLevelErrorMasker,
        active_difficulty: Optional[str] = None,
        max_length: int = 2048,
    ):
        self.masker = masker
        self.active_difficulty = active_difficulty
        self.max_length = max_length

        if active_difficulty:
            self.trajectories = [t for t in trajectories if t.difficulty == active_difficulty]
        else:
            # Order by curriculum: Easy -> Medium -> Hard
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            self.trajectories = sorted(
                trajectories, key=lambda t: diff_order.get(t.difficulty.lower(), 1)
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.trajectories[idx]
        return self.masker.format_and_mask_trajectory(
            system_prompt=item.system_prompt,
            steps=item.steps,
            final_solution=item.final_patch,
            max_length=self.max_length,
        )

    @classmethod
    def create_synthetic_curriculum(cls, masker: StepLevelErrorMasker) -> CurriculumTrajectoryDataset:
        """Create a synthetic curriculum dataset demonstrating multi-turn trajectories across difficulty levels."""
        from src.data.sri_formatter import SRIFormatter

        trajectories = [
            # Easy: 1 turn, localized single-file fix
            CuratedTrajectory(
                instance_id="easy_arithmetic_fix",
                difficulty="easy",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"calc.py\" start_line=\"1\" end_line=\"10\" />",
                        observation_text="def add(a, b):\n    return a - b",
                        is_error=False,
                    )
                ],
                final_patch=SRIFormatter.format_block("calc.py", "    return a - b", "    return a + b"),
            ),
            # Medium: 3 turns, error recovery and test execution
            CuratedTrajectory(
                instance_id="medium_connection_pool_fix",
                difficulty="medium",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"run_test\" command=\"pytest tests/test_pool.py\" />",
                        observation_text="FAILED tests/test_pool.py::test_reconnect - TimeoutError: connection dropped",
                        is_error=True,  # Execution error: masked!
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"pool.py\" start_line=\"40\" end_line=\"60\" />",
                        observation_text="45 | def reconnect(self):\n46 |     if not self.closed:\n47 |         pass",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"search_code\" query=\"self.closed\" />",
                        observation_text="pool.py:46:    if not self.closed:\npool.py:80:        self.closed = True",
                        is_error=False,
                    ),
                ],
                final_patch=SRIFormatter.format_block(
                    "pool.py",
                    "    if not self.closed:\n        pass",
                    "    if not self.closed:\n        self.connect()",
                ),
            ),
            # Hard: Multi-step cross-module refactor
            CuratedTrajectory(
                instance_id="hard_django_query_fix",
                difficulty="hard",
                system_prompt="You are an autonomous SWE agent. Fix bugs using SRI format.",
                steps=[
                    TrajectoryStep(
                        action_text="<tool_call name=\"search_code\" query=\"FilteredRelation\" />",
                        observation_text="django/db/models/sql/query.py:120: class FilteredRelation",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"view_file\" path=\"django/db/models/sql/query.py\" start_line=\"120\" end_line=\"150\" />",
                        observation_text="125 | def resolve_expression(self):\n126 |     return self.relation_name",
                        is_error=False,
                    ),
                    TrajectoryStep(
                        action_text="<tool_call name=\"run_test\" command=\"pytest tests/queries/test_filtered_relation.py\" />",
                        observation_text="FAILED tests/queries/test_filtered_relation.py::test_aggregate - AttributeError: 'NoneType' object has no attribute 'name'",
                        is_error=True,  # Masked!
                    ),
                ],
                final_patch=SRIFormatter.format_block(
                    "django/db/models/sql/query.py",
                    "    return self.relation_name",
                    "    return self.relation_name if self.relation_name else None",
                ),
            ),
        ]

        return cls(trajectories=trajectories, masker=masker)

    @classmethod
    def from_jsonl(
        cls,
        file_path: str,
        masker: StepLevelErrorMasker,
        active_difficulty: Optional[str] = None,
        max_length: int = 2048,
    ) -> CurriculumTrajectoryDataset:
        """Load trajectories from an external JSON Lines dataset file."""
        import json
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        trajectories: List[CuratedTrajectory] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                steps = [
                    TrajectoryStep(
                        action_text=s["action_text"],
                        observation_text=s["observation_text"],
                        is_error=s.get("is_error", False),
                    )
                    for s in data.get("steps", [])
                ]
                trajectories.append(
                    CuratedTrajectory(
                        instance_id=data["instance_id"],
                        difficulty=data.get("difficulty", "medium"),
                        system_prompt=data.get("system_prompt", "You are an autonomous SWE agent. Fix bugs using SRI format."),
                        steps=steps,
                        final_patch=data.get("final_patch", ""),
                    )
                )

        return cls(
            trajectories=trajectories,
            masker=masker,
            active_difficulty=active_difficulty,
            max_length=max_length,
        )

    @classmethod
    def generate_expanded_curriculum(
        cls,
        masker: StepLevelErrorMasker,
        active_difficulty: Optional[str] = None,
    ) -> CurriculumTrajectoryDataset:
        """Generate a 30-instance diverse curriculum covering real repository bug patterns."""
        from src.data.sri_formatter import SRIFormatter

        sys_prompt = "You are an expert autonomous SWE agent. Resolve the bug using Search-and-Replace Infilling (SRI)."
        trajectories: List[CuratedTrajectory] = []

        # ==========================================
        # 10 Easy Trajectories: 1-2 turns, localized guards
        # ==========================================
        easy_templates = [
            ("easy_calc_zero_div", "calc.py", "def divide(a, b):\n    return a / b", "def divide(a, b):\n    if b == 0:\n        return float('inf')\n    return a / b"),
            ("easy_dict_get_fallback", "config.py", "def get_timeout(cfg):\n    return cfg['timeout']", "def get_timeout(cfg):\n    return cfg.get('timeout', 30)"),
            ("easy_list_index_guard", "pager.py", "def get_page(items, p):\n    return items[p]", "def get_page(items, p):\n    return items[p] if 0 <= p < len(items) else None"),
            ("easy_none_attribute", "auth.py", "def get_role(user):\n    return user.role.name", "def get_role(user):\n    return user.role.name if user and user.role else 'guest'"),
            ("easy_str_decode_utf8", "parser.py", "def to_str(b):\n    return b.decode('ascii')", "def to_str(b):\n    return b.decode('utf-8', errors='replace')"),
            ("easy_path_exists_check", "storage.py", "def load_file(p):\n    return open(p).read()", "def load_file(p):\n    import os\n    return open(p).read() if os.path.exists(p) else ''"),
            ("easy_int_conversion", "converter.py", "def parse_num(v):\n    return int(v)", "def parse_num(v):\n    try:\n        return int(v)\n    except (ValueError, TypeError):\n        return 0"),
            ("easy_float_nan_guard", "stats.py", "def average(arr):\n    return sum(arr) / len(arr)", "def average(arr):\n    return sum(arr) / len(arr) if arr else 0.0"),
            ("easy_astropy_unit_check", "units.py", "def ensure_meters(q):\n    return q.to('m')", "def ensure_meters(q):\n    return q.to('m') if hasattr(q, 'to') else q"),
            ("easy_set_remove_discard", "session.py", "def drop_token(s, t):\n    s.remove(t)", "def drop_token(s, t):\n    s.discard(t)"),
        ]

        for i_id, path, old_c, new_c in easy_templates:
            trajectories.append(
                CuratedTrajectory(
                    instance_id=i_id,
                    difficulty="easy",
                    system_prompt=sys_prompt,
                    steps=[
                        TrajectoryStep(
                            action_text=f'<tool_call name="view_file" path="{path}" start_line="1" end_line="15" />',
                            observation_text=f"=== File: {path} ===\n{old_c}",
                            is_error=False,
                        )
                    ],
                    final_patch=SRIFormatter.format_block(path, old_c, new_c),
                )
            )

        # ==========================================
        # 10 Medium Trajectories: 2-3 turns with exploration & masked blunder
        # ==========================================
        medium_templates = [
            ("med_conn_pool_reconnect", "pool.py", "def acquire(self):\n    return self.conns.pop()", "def acquire(self):\n    if not self.conns:\n        self.spawn()\n    return self.conns.pop()"),
            ("med_json_date_serializer", "json_util.py", "def serialize(obj):\n    return json.dumps(obj)", "def serialize(obj):\n    return json.dumps(obj, default=str)"),
            ("med_requests_backoff", "client.py", "def request_url(url):\n    return requests.get(url)", "def request_url(url, retries=3):\n    for _ in range(retries):\n        try:\n            return requests.get(url)\n        except requests.RequestException:\n            pass\n    return None"),
            ("med_sql_param_binding", "db.py", "def search_user(cur, name):\n    return cur.execute(f'SELECT * FROM u WHERE name = {name}')", "def search_user(cur, name):\n    return cur.execute('SELECT * FROM u WHERE name = ?', (name,))"),
            ("med_asyncio_cancel_cleanup", "runner.py", "async def fetch():\n    await task", "async def fetch():\n    try:\n        await task\n    except asyncio.CancelledError:\n        await cleanup()"),
            ("med_mutable_default_arg", "schema.py", "def init_schema(fields=[]):\n    self.fields = fields", "def init_schema(fields=None):\n    self.fields = fields if fields is not None else []"),
            ("med_context_suppress_exc", "manager.py", "def __exit__(self, *args):\n    return False", "def __exit__(self, exc_type, *args):\n    return exc_type is not None"),
            ("med_symlink_traversal", "sandbox.py", "def resolve_path(p):\n    return os.path.abspath(p)", "def resolve_path(p, root):\n    resolved = os.path.realpath(p)\n    if not resolved.startswith(os.path.realpath(root)):\n        raise PermissionError('Symlink traversal')\n    return resolved"),
            ("med_astropy_table_col", "table.py", "def add_col(t, c):\n    t[c.name] = c", "def add_col(t, c):\n    if c.name in t.colnames:\n        raise ValueError(f'Column {c.name} already exists')\n    t[c.name] = c"),
            ("med_markdown_link_edge", "md.py", "def parse_link(txt):\n    return re.findall(r'\\[(.*?)\\]\\((.*?)\\)', txt)", "def parse_link(txt):\n    return re.findall(r'\\[([^\\]]+)\\]\\(([^\\)]+)\\)', txt)"),
        ]

        for i_id, path, old_c, new_c in medium_templates:
            trajectories.append(
                CuratedTrajectory(
                    instance_id=i_id,
                    difficulty="medium",
                    system_prompt=sys_prompt,
                    steps=[
                        TrajectoryStep(
                            action_text=f'<tool_call name="search_code" query="{path.split(".")[0]}" />',
                            observation_text=f"{path}:10: {old_c.splitlines()[0]}",
                            is_error=False,
                        ),
                        TrajectoryStep(
                            action_text=f'<tool_call name="run_test" command="pytest tests/test_{path}" />',
                            observation_text=f"FAILED tests/test_{path}::test_edge - AssertionFailed",
                            is_error=True,  # Execution blunder: masked!
                        ),
                        TrajectoryStep(
                            action_text=f'<tool_call name="view_file" path="{path}" start_line="1" end_line="20" />',
                            observation_text=f"=== File: {path} ===\n{old_c}",
                            is_error=False,
                        ),
                    ],
                    final_patch=SRIFormatter.format_block(path, old_c, new_c),
                )
            )

        # ==========================================
        # 10 Hard Trajectories: 3-4 turns across cross-module refactors
        # ==========================================
        hard_templates = [
            ("hard_django_filtered_relation", "django/db/models/sql/query.py", "def resolve_expression(self):\n    return self.relation_name", "def resolve_expression(self):\n    return self.relation_name if self.relation_name else None"),
            ("hard_astropy_separable_matrix", "astropy/modeling/separable.py", "def separability_matrix(model):\n    return _separable(model)", "def separability_matrix(model):\n    if hasattr(model, '_calculate_separability_matrix'):\n        return model._calculate_separability_matrix()\n    return _separable(model)"),
            ("hard_scipy_sparse_crs_multiply", "scipy/sparse/linalg.py", "def matmul(A, B):\n    return A.dot(B)", "def matmul(A, B):\n    if A.shape[1] != B.shape[0]:\n        raise ValueError(f'Dimension mismatch: {A.shape} vs {B.shape}')\n    return A.dot(B)"),
            ("hard_sympy_integral_singularity", "sympy/integrals/risch.py", "def integrate_term(term):\n    return term.integrate()", "def integrate_term(term):\n    if term.is_singular:\n        return term.principal_value()\n    return term.integrate()"),
            ("hard_flask_route_collision", "flask/blueprints.py", "def register_route(app, endpoint):\n    app.view_functions[endpoint] = fn", "def register_route(app, endpoint):\n    if endpoint in app.view_functions:\n        endpoint = f'{self.name}.{endpoint}'\n    app.view_functions[endpoint] = fn"),
            ("hard_thread_safe_lru_invalidation", "cache/lru.py", "def invalidate(self, k):\n    del self.data[k]", "def invalidate(self, k):\n    with self._lock:\n        self.data.pop(k, None)"),
            ("hard_nested_schema_aggregator", "schema/validator.py", "def validate_tree(node):\n    return node.check()", "def validate_tree(node, errors=None):\n    errs = errors if errors is not None else []\n    if not node.check():\n        errs.append(node.path)\n    for c in node.children:\n        validate_tree(c, errs)\n    return errs"),
            ("hard_circular_lazy_loader", "core/loader.py", "from sub.worker import Worker", "def get_worker():\n    from sub.worker import Worker\n    return Worker()"),
            ("hard_worker_deadlock_hierarchy", "concurrency/pool.py", "def swap(p1, p2):\n    p1.lock.acquire()\n    p2.lock.acquire()", "def swap(p1, p2):\n    first, second = (p1, p2) if id(p1) < id(p2) else (p2, p1)\n    with first.lock, second.lock:\n        first.balance, second.balance = second.balance, first.balance"),
            ("hard_worktree_orphan_lifecycle", "git/worktree.py", "def prune_all(root):\n    subprocess.run(['git', 'worktree', 'prune'])", "def prune_all(root):\n    subprocess.run(['git', 'worktree', 'prune', '--expire=now'], check=True)"),
        ]

        for i_id, path, old_c, new_c in hard_templates:
            trajectories.append(
                CuratedTrajectory(
                    instance_id=i_id,
                    difficulty="hard",
                    system_prompt=sys_prompt,
                    steps=[
                        TrajectoryStep(
                            action_text=f'<tool_call name="search_code" query="{path.split("/")[-1].split(".")[0]}" />',
                            observation_text=f"{path}:1: def {old_c.splitlines()[0].split('(')[0].replace('def ', '')}",
                            is_error=False,
                        ),
                        TrajectoryStep(
                            action_text=f'<tool_call name="view_file" path="{path}" start_line="1" end_line="35" />',
                            observation_text=f"=== File: {path} ===\n{old_c}",
                            is_error=False,
                        ),
                        TrajectoryStep(
                            action_text=f'<tool_call name="run_test" command="pytest tests/test_{path.split("/")[-1]}" />',
                            observation_text=f"FAILED tests/test_{path.split('/')[-1]} - Deadlock / Regression detected",
                            is_error=True,  # Execution blunder: masked!
                        ),
                        TrajectoryStep(
                            action_text=f'<tool_call name="view_file" path="{path}" start_line="1" end_line="50" />',
                            observation_text=f"=== File: {path} (Context Confirmed) ===\n{old_c}",
                            is_error=False,
                        ),
                    ],
                    final_patch=SRIFormatter.format_block(path, old_c, new_c),
                )
            )

        return cls(trajectories=trajectories, masker=masker, active_difficulty=active_difficulty)
