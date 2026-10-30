"""RepoGraph Engine: AST & Call-Graph Analysis for Codebases.

Builds a directed graph of repository elements (Modules, Classes, Functions, Methods),
extracts dependencies (CALLS, IMPORTS, INHERITS_FROM, DEFINES), computes structural
node embeddings, and determines the blast-radius of breaking changes.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx


@dataclass
class CodeNode:
    """Represents a node in the repository code graph."""

    id: str  # Unique identifier (e.g., 'src/api/pay.py::process_payment')
    name: str  # Symbol name (e.g., 'process_payment')
    file_path: str  # Relative file path
    node_type: str  # 'module', 'class', 'function', 'method'
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    args: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None  # Enclosing class or module
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlastRadiusItem:
    """An affected node in the blast radius of a code modification."""

    node_id: str
    node_name: str
    file_path: str
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    distance: int  # Hops from modified node in call graph
    reason: str  # Explanation of why this caller is broken or affected
    call_line_numbers: List[int] = field(default_factory=list)
    call_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeGraphEngine:
    """Parses Python source files into a NetworkX directed dependency graph."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, CodeNode] = {}
        self.file_contents: Dict[str, str] = {}

    def add_file(self, file_path: str, content: str) -> None:
        """Parse a Python source file and add its AST nodes to the graph."""
        clean_path = str(Path(file_path)).replace("\\", "/")
        self.file_contents[clean_path] = content

        module_node_id = f"file::{clean_path}"
        mod_node = CodeNode(
            id=module_node_id,
            name=Path(clean_path).stem,
            file_path=clean_path,
            node_type="module",
            line_start=1,
            line_end=len(content.splitlines()) or 1,
        )
        self.nodes[module_node_id] = mod_node
        self.graph.add_node(module_node_id, **mod_node.to_dict())

        try:
            tree = ast.parse(content, filename=clean_path)
        except SyntaxError:
            return

        self._extract_ast_definitions(tree, clean_path, module_node_id)
        self._extract_ast_relationships(tree, clean_path, module_node_id)

    def _extract_ast_definitions(
        self, tree: ast.AST, file_path: str, module_id: str
    ) -> None:
        """Extract classes, functions, and methods into the node registry."""
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                class_id = f"{file_path}::{stmt.name}"
                c_node = CodeNode(
                    id=class_id,
                    name=stmt.name,
                    file_path=file_path,
                    node_type="class",
                    line_start=stmt.lineno,
                    line_end=stmt.end_lineno or stmt.lineno,
                    parent_id=module_id,
                    docstring=ast.get_docstring(stmt) or "",
                )
                self.nodes[class_id] = c_node
                self.graph.add_node(class_id, **c_node.to_dict())
                self.graph.add_edge(module_id, class_id, relation="DEFINES")

                # Extract inheritance
                for base in stmt.bases:
                    if isinstance(base, ast.Name):
                        self.graph.add_edge(
                            class_id, base.id, relation="INHERITS_FROM"
                        )

                # Extract methods
                for item in stmt.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{file_path}::{stmt.name}.{item.name}"
                        arg_names = [a.arg for a in item.args.args]
                        m_node = CodeNode(
                            id=method_id,
                            name=f"{stmt.name}.{item.name}",
                            file_path=file_path,
                            node_type="method",
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            args=arg_names,
                            parent_id=class_id,
                            docstring=ast.get_docstring(item) or "",
                        )
                        self.nodes[method_id] = m_node
                        self.graph.add_node(method_id, **m_node.to_dict())
                        self.graph.add_edge(
                            class_id, method_id, relation="DEFINES"
                        )

            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_id = f"{file_path}::{stmt.name}"
                arg_names = [a.arg for a in stmt.args.args]
                f_node = CodeNode(
                    id=fn_id,
                    name=stmt.name,
                    file_path=file_path,
                    node_type="function",
                    line_start=stmt.lineno,
                    line_end=stmt.end_lineno or stmt.lineno,
                    args=arg_names,
                    parent_id=module_id,
                    docstring=ast.get_docstring(stmt) or "",
                )
                self.nodes[fn_id] = f_node
                self.graph.add_node(fn_id, **f_node.to_dict())
                self.graph.add_edge(module_id, fn_id, relation="DEFINES")

    def _extract_ast_relationships(
        self, tree: ast.AST, file_path: str, module_id: str
    ) -> None:
        """Extract function call invocations and imports to create CALLS/IMPORTS edges."""

        class RelationshipVisitor(ast.NodeVisitor):
            def __init__(self, engine: CodeGraphEngine, current_file: str):
                self.engine = engine
                self.current_file = current_file
                self.current_scope = module_id

            def visit_ClassDef(self, node: ast.ClassDef):
                old_scope = self.current_scope
                self.current_scope = f"{self.current_file}::{node.name}"
                self.generic_visit(node)
                self.current_scope = old_scope

            def visit_FunctionDef(self, node: ast.FunctionDef):
                old_scope = self.current_scope
                if "::" in self.current_scope and not self.current_scope.startswith("file::"):
                    cls_name = self.current_scope.split("::")[1]
                    self.current_scope = f"{self.current_file}::{cls_name}.{node.name}"
                else:
                    self.current_scope = f"{self.current_file}::{node.name}"
                self.generic_visit(node)
                self.current_scope = old_scope

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call):
                target_name = None
                if isinstance(node.func, ast.Name):
                    target_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    target_name = node.func.attr

                if target_name:
                    matched_nodes = [
                        nid
                        for nid, n in self.engine.nodes.items()
                        if n.name == target_name or n.name.endswith(f".{target_name}")
                    ]
                    for target_id in matched_nodes:
                        if target_id != self.current_scope:
                            self.engine.graph.add_edge(
                                self.current_scope,
                                target_id,
                                relation="CALLS",
                                lineno=node.lineno,
                            )
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    for nid in self.engine.nodes:
                        if nid.startswith("file::") and alias.name in nid:
                            self.engine.graph.add_edge(
                                module_id, nid, relation="IMPORTS"
                            )
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom):
                mod_name = node.module or ""
                for alias in node.names:
                    target_suffix = f"{alias.name}"
                    for nid, n in self.engine.nodes.items():
                        if n.name == target_suffix or (mod_name and mod_name in n.file_path):
                            self.engine.graph.add_edge(
                                module_id, nid, relation="IMPORTS"
                            )
                self.generic_visit(node)

        visitor = RelationshipVisitor(self, file_path)
        visitor.visit(tree)

    def compute_node_embeddings(self) -> None:
        """Compute structural, topological, and lexical embeddings for each node.

        Returns an 8-dimensional normalized embedding combining:
        - Structural role (module=0, class=1, function=2, method=3)
        - In-degree (how heavily this node is depended upon)
        - Out-degree (how complex are this node's external calls)
        - Line span / complexity
        - 4 hashed lexical tokens from name and signature
        """
        role_map = {"module": 0.0, "class": 0.33, "function": 0.66, "method": 1.0}

        for node_id, node in self.nodes.items():
            in_deg = self.graph.in_degree(node_id) if self.graph.has_node(node_id) else 0
            out_deg = self.graph.out_degree(node_id) if self.graph.has_node(node_id) else 0
            line_span = max(1, node.line_end - node.line_start + 1)

            h = hashlib.sha256(node.name.encode("utf-8")).digest()
            lex1 = (h[0] / 127.5) - 1.0
            lex2 = (h[1] / 127.5) - 1.0
            lex3 = (h[2] / 127.5) - 1.0
            lex4 = (h[3] / 127.5) - 1.0

            raw_vec = [
                role_map.get(node.node_type, 0.5),
                min(1.0, math.log1p(in_deg) / 3.0),
                min(1.0, math.log1p(out_deg) / 3.0),
                min(1.0, math.log1p(line_span) / 5.0),
                lex1,
                lex2,
                lex3,
                lex4,
            ]
            norm = math.sqrt(sum(x * x for x in raw_vec)) or 1.0
            node.embedding = [round(x / norm, 4) for x in raw_vec]

    def compute_blast_radius(
        self,
        modified_node_name_or_id: str,
        breaking_change_description: str = "",
        max_hops: int = 4,
    ) -> List[BlastRadiusItem]:
        """Compute the blast radius of a breaking code change.

        Finds all downstream callers that transitively invoke or depend on the modified
        node using the reverse call-graph DAG and ranks them by severity.
        """
        target_id = None
        for nid, n in self.nodes.items():
            if nid == modified_node_name_or_id or n.name == modified_node_name_or_id:
                target_id = nid
                break
            if modified_node_name_or_id in nid:
                target_id = nid

        if not target_id or not self.graph.has_node(target_id):
            return []

        target_node = self.nodes[target_id]
        blast_items: List[BlastRadiusItem] = []
        visited: Set[str] = {target_id}

        queue: List[Tuple[str, int]] = [(target_id, 0)]

        while queue:
            curr_id, distance = queue.pop(0)
            if distance >= max_hops:
                continue

            for predecessor in self.graph.predecessors(curr_id):
                edge_data = self.graph.get_edge_data(predecessor, curr_id)
                is_call_or_import = False
                call_lines = []
                for _, attr in (edge_data or {}).items():
                    rel = attr.get("relation")
                    if rel in ("CALLS", "IMPORTS", "INHERITS_FROM"):
                        is_call_or_import = True
                        if "lineno" in attr:
                            call_lines.append(attr["lineno"])

                if is_call_or_import and predecessor not in visited:
                    visited.add(predecessor)
                    pred_node = self.nodes.get(predecessor)
                    if pred_node and pred_node.node_type in ("function", "method", "class"):
                        if distance == 0:
                            severity = "CRITICAL"
                            reason = (
                                f"Directly invokes `{target_node.name}`. "
                                f"Signature mismatch or argument arity change directly impacts this call."
                            )
                        elif distance == 1:
                            severity = "WARNING"
                            reason = (
                                f"Transitive caller (2 hops) via `{self.nodes.get(curr_id, target_node).name}`. "
                                f"May propagate invalid return types or missing parameters."
                            )
                        else:
                            severity = "INFO"
                            reason = f"Upstream consumer ({distance + 1} hops removed)."

                        snippet = ""
                        content = self.file_contents.get(pred_node.file_path, "")
                        if content and call_lines:
                            lines = content.splitlines()
                            target_line = call_lines[0] - 1
                            start = max(0, target_line - 1)
                            end = min(len(lines), target_line + 2)
                            snippet = "\n".join(lines[start:end])

                        blast_items.append(
                            BlastRadiusItem(
                                node_id=pred_node.id,
                                node_name=pred_node.name,
                                file_path=pred_node.file_path,
                                severity=severity,
                                distance=distance + 1,
                                reason=reason,
                                call_line_numbers=call_lines,
                                call_snippet=snippet,
                            )
                        )
                        queue.append((predecessor, distance + 1))

        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        blast_items.sort(key=lambda x: (severity_order.get(x.severity, 3), x.distance))
        return blast_items

    def export_graph_for_visualization(
        self,
        highlight_node_id: Optional[str] = None,
        blast_radius: Optional[List[BlastRadiusItem]] = None,
    ) -> Dict[str, Any]:
        """Export nodes and edges formatted for interactive D3/SVG canvas rendering."""
        blast_lookup = {item.node_id: item for item in (blast_radius or [])}

        viz_nodes = []
        for nid, n in self.nodes.items():
            status = "normal"
            if nid == highlight_node_id or (
                highlight_node_id and n.name == highlight_node_id
            ):
                status = "target"
            elif nid in blast_lookup:
                status = (
                    "critical"
                    if blast_lookup[nid].severity == "CRITICAL"
                    else "warning"
                )

            viz_nodes.append({
                "id": nid,
                "name": n.name,
                "file_path": n.file_path,
                "node_type": n.node_type,
                "lines": f"L{n.line_start}-{n.line_end}",
                "args": n.args,
                "status": status,
                "severity": blast_lookup.get(nid).severity if nid in blast_lookup else None,
                "reason": blast_lookup.get(nid).reason if nid in blast_lookup else None,
                "embedding": n.embedding,
            })

        viz_edges = []
        for u, v, k, attr in self.graph.edges(keys=True, data=True):
            viz_edges.append({
                "source": u,
                "target": v,
                "relation": attr.get("relation", "DEPENDS"),
                "lineno": attr.get("lineno"),
            })

        return {
            "nodes": viz_nodes,
            "edges": viz_edges,
            "total_nodes": len(viz_nodes),
            "total_edges": len(viz_edges),
        }
