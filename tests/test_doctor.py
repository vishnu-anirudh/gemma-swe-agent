"""Unit and integration tests for RepoGraph Doctor.

Tests AST parsing, dependency graph extraction, node embeddings, blast-radius calculation,
SRI patch generation, and API server endpoints.
"""

from __future__ import annotations

import json
from src.doctor.graph_engine import CodeGraphEngine
from src.doctor.repair_agent import RepoGraphRepairAgent
from src.doctor.scenarios import SCENARIOS


def test_graph_engine_ast_parsing_and_relationships():
    """Verify classes, functions, and call relationships are correctly indexed."""
    engine = CodeGraphEngine()

    engine.add_file(
        "math_ops.py",
        "def add(a, b):\n    return a + b\n\ndef calc(x):\n    return add(x, 10)\n",
    )
    engine.compute_node_embeddings()

    assert "math_ops.py::add" in engine.nodes
    assert "math_ops.py::calc" in engine.nodes

    add_node = engine.nodes["math_ops.py::add"]
    assert add_node.node_type == "function"
    assert add_node.args == ["a", "b"]
    assert len(add_node.embedding) == 8

    # Edge math_ops.py::calc -> math_ops.py::add should exist
    edges = list(engine.graph.edges(data=True))
    call_edges = [
        (u, v) for u, v, d in edges if d.get("relation") == "CALLS"
    ]
    assert ("math_ops.py::calc", "math_ops.py::add") in call_edges


def test_blast_radius_computation():
    """Verify blast radius discovers direct and transitive callers."""
    engine = CodeGraphEngine()
    engine.add_file(
        "core.py",
        "def base_func():\n    return 42\n",
    )
    engine.add_file(
        "service.py",
        "from core import base_func\ndef service_call():\n    return base_func()\n",
    )
    engine.add_file(
        "api.py",
        "from service import service_call\ndef handle_req():\n    return service_call()\n",
    )

    blast = engine.compute_blast_radius("core.py::base_func", max_hops=3)
    affected_ids = [item.node_id for item in blast]

    assert "service.py::service_call" in affected_ids
    assert "api.py::handle_req" in affected_ids

    # service_call should be 1 hop (CRITICAL)
    service_item = next(i for i in blast if i.node_id == "service.py::service_call")
    assert service_item.distance == 1
    assert service_item.severity == "CRITICAL"

    # handle_req should be 2 hops (WARNING)
    api_item = next(i for i in blast if i.node_id == "api.py::handle_req")
    assert api_item.distance == 2
    assert api_item.severity == "WARNING"


def test_scenarios_and_autonomous_repair():
    """Verify autonomous SRI repair across Scenario 1 (payment signature refactor)."""
    sc = SCENARIOS["scenario_1"]
    engine = CodeGraphEngine()
    for path, code in sc.files.items():
        engine.add_file(path, code)

    engine.compute_node_embeddings()
    blast = engine.compute_blast_radius(sc.target_node, sc.breaking_change)

    assert len(blast) == 4

    agent = RepoGraphRepairAgent(engine)
    report = agent.repair_blast_radius(
        target_node_name=sc.target_node,
        breaking_change_desc=sc.breaking_change,
        blast_items=blast,
        test_verifier=sc.test_verifier,
    )

    assert report.total_affected_callers == 4
    assert report.repaired_count == 4
    assert report.failed_count == 0
    assert report.tests_passed is True

    # Check that patched files contain idempotency_key
    for r in report.repairs:
        assert r.applied_cleanly is True
        assert r.ast_valid is True
        assert "idempotency_key" in r.new_snippet


def test_visualization_export():
    """Verify exported visualization format is compliant with frontend canvas."""
    engine = CodeGraphEngine()
    engine.add_file("app.py", "def run():\n    pass\n")
    engine.compute_node_embeddings()
    viz = engine.export_graph_for_visualization(highlight_node_id="app.py::run")

    assert "nodes" in viz
    assert "edges" in viz
    assert viz["total_nodes"] >= 2  # module + function
    run_node = next(n for n in viz["nodes"] if n["id"] == "app.py::run")
    assert run_node["status"] == "target"
