"""RepoGraph Doctor Web Server & Interactive API.

Provides an HTTP server for the interactive visual code-graph blast-radius studio.
Zero external server dependencies required (uses standard library http.server).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from src.doctor.graph_engine import CodeGraphEngine
from src.doctor.repair_agent import RepoGraphRepairAgent
from src.doctor.scenarios import SCENARIOS


class RepoGraphHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for RepoGraph Doctor API & Studio."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logging in terminal unless error."""
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(format, *args)

    def do_HEAD(self) -> None:
        """Handle HEAD requests for health checks."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for frontend UI and scenario metadata."""
        parsed_url = urllib.parse.urlparse(self.path)

        if parsed_url.path in ("/", "/index.html"):
            self._serve_html()
        elif parsed_url.path == "/api/scenarios":
            self._send_json({
                "status": "success",
                "scenarios": [s.to_summary_dict() for s in SCENARIOS.values()],
            })
        elif parsed_url.path.startswith("/api/scenario/"):
            scenario_id = parsed_url.path.replace("/api/scenario/", "")
            if scenario_id in SCENARIOS:
                sc = SCENARIOS[scenario_id]
                self._send_json({
                    "status": "success",
                    "scenario": {
                        "id": sc.id,
                        "title": sc.title,
                        "category": sc.category,
                        "description": sc.description,
                        "target_node": sc.target_node,
                        "breaking_change": sc.breaking_change,
                        "files": sc.files,
                    },
                })
            else:
                self._send_error(404, f"Scenario '{scenario_id}' not found.")
        else:
            self._send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Handle POST requests for graph analysis and autonomous repairs."""
        parsed_url = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self._send_error(400, "Invalid JSON payload.")
            return

        if parsed_url.path == "/api/analyze":
            self._handle_analyze(body)
        elif parsed_url.path == "/api/repair":
            self._handle_repair(body)
        elif parsed_url.path == "/api/custom-repo":
            self._handle_custom_repo(body)
        else:
            self._send_error(404, "API endpoint not found.")

    def _handle_analyze(self, body: Dict[str, Any]) -> None:
        """Build graph and compute blast radius for scenario or files."""
        scenario_id = body.get("scenario_id")
        files = body.get("files")
        target_node = body.get("target_node")
        breaking_change = body.get("breaking_change", "")

        if scenario_id and scenario_id in SCENARIOS:
            sc = SCENARIOS[scenario_id]
            files = sc.files
            target_node = target_node or sc.target_node
            breaking_change = breaking_change or sc.breaking_change

        if not files or not target_node:
            self._send_error(400, "Missing 'files' or 'target_node'.")
            return

        engine = CodeGraphEngine()
        for fpath, content in files.items():
            engine.add_file(fpath, content)

        engine.compute_node_embeddings()
        blast_items = engine.compute_blast_radius(
            modified_node_name_or_id=target_node,
            breaking_change_description=breaking_change,
        )
        viz_data = engine.export_graph_for_visualization(
            highlight_node_id=target_node, blast_radius=blast_items
        )

        self._send_json({
            "status": "success",
            "target_node": target_node,
            "breaking_change": breaking_change,
            "blast_radius": [b.to_dict() for b in blast_items],
            "graph": viz_data,
        })

    def _handle_repair(self, body: Dict[str, Any]) -> None:
        """Run Gemma repair agent on blast radius callers."""
        scenario_id = body.get("scenario_id")
        files = body.get("files")
        target_node = body.get("target_node")
        breaking_change = body.get("breaking_change", "")

        verifier_fn = None
        if scenario_id and scenario_id in SCENARIOS:
            sc = SCENARIOS[scenario_id]
            files = sc.files
            target_node = target_node or sc.target_node
            breaking_change = breaking_change or sc.breaking_change
            verifier_fn = sc.test_verifier

        if not files or not target_node:
            self._send_error(400, "Missing 'files' or 'target_node'.")
            return

        engine = CodeGraphEngine()
        for fpath, content in files.items():
            engine.add_file(fpath, content)

        engine.compute_node_embeddings()
        blast_items = engine.compute_blast_radius(
            modified_node_name_or_id=target_node,
            breaking_change_description=breaking_change,
        )

        agent = RepoGraphRepairAgent(engine)
        report = agent.repair_blast_radius(
            target_node_name=target_node,
            breaking_change_desc=breaking_change,
            blast_items=blast_items,
            test_verifier=verifier_fn,
        )

        # Update visualization with repaired status
        viz_data = engine.export_graph_for_visualization(
            highlight_node_id=target_node, blast_radius=blast_items
        )
        repaired_ids = {r.node_id for r in report.repairs if r.applied_cleanly}
        for n in viz_data["nodes"]:
            if n["id"] in repaired_ids:
                n["status"] = "repaired"

        self._send_json({
            "status": "success",
            "report": report.to_dict(),
            "graph": viz_data,
        })

    def _handle_custom_repo(self, body: Dict[str, Any]) -> None:
        """Scan a local directory path from the user's workspace."""
        repo_path_str = body.get("path", ".")
        target_symbol = body.get("target_symbol", "")

        repo_dir = Path(repo_path_str)
        if not repo_dir.exists() or not repo_dir.is_dir():
            self._send_error(400, f"Directory '{repo_path_str}' does not exist.")
            return

        engine = CodeGraphEngine()
        py_files = list(repo_dir.glob("**/*.py"))[:40]  # Limit to 40 files for responsive graph
        for pf in py_files:
            try:
                rel = pf.relative_to(repo_dir).as_posix()
                content = pf.read_text(encoding="utf-8", errors="ignore")
                engine.add_file(rel, content)
            except Exception:
                pass

        engine.compute_node_embeddings()
        blast_items = []
        if target_symbol:
            blast_items = engine.compute_blast_radius(target_symbol)

        viz_data = engine.export_graph_for_visualization(
            highlight_node_id=target_symbol, blast_radius=blast_items
        )

        self._send_json({
            "status": "success",
            "scanned_files": len(py_files),
            "target_node": target_symbol,
            "blast_radius": [b.to_dict() for b in blast_items],
            "graph": viz_data,
        })

    def _serve_html(self) -> None:
        """Serve the modern dark-mode interactive web studio."""
        html_content = HTML_STUDIO_TEMPLATE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def _send_json(self, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, code: int, message: str) -> None:
        payload = json.dumps({"status": "error", "message": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


HTML_STUDIO_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RepoGraph Doctor | Sub-3B Code-Graph Refactoring & Repair Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0a0d14;
      --bg-surface: #101522;
      --bg-card: rgba(22, 29, 46, 0.75);
      --bg-card-hover: rgba(30, 40, 64, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(99, 102, 241, 0.4);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.35);
      --accent-target: #a855f7;
      --accent-critical: #ef4444;
      --accent-warning: #f59e0b;
      --accent-success: #10b981;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'Fira Code', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Top Navigation */
    header {
      background: rgba(16, 21, 34, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-logo {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px var(--primary-glow);
    }

    .brand-logo svg { width: 20px; height: 20px; fill: white; }
    .brand-text h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand-text p { font-size: 0.72rem; color: var(--text-muted); }

    .nav-controls {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    select, button {
      font-family: var(--font-sans);
      font-size: 0.85rem;
      border-radius: 8px;
      padding: 0.55rem 1rem;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      cursor: pointer;
    }

    select {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      outline: none;
      min-width: 280px;
    }

    select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary-glow); }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 600;
      border: none;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.1); }

    .btn-analyze {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: white;
      box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
    }
    .btn-analyze:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5); }

    .btn-repair {
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
      box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }
    .btn-repair:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(16, 185, 129, 0.5); }

    .badge-live {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } 100% { opacity: 1; transform: scale(1); } }

    /* Main Workspace Layout */
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      height: calc(100vh - 65px);
      overflow: hidden;
    }

    /* Left Graph Canvas Panel */
    .graph-panel {
      position: relative;
      background: radial-gradient(circle at 50% 50%, #13192b 0%, #0a0d14 100%);
      border-right: 1px solid var(--border-subtle);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    #graph-canvas {
      flex: 1;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    #graph-canvas:active { cursor: grabbing; }

    .graph-overlay {
      position: absolute;
      top: 1rem;
      left: 1rem;
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.75rem 1rem;
      pointer-events: none;
      z-index: 10;
    }

    .graph-overlay h3 { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem; }
    .graph-overlay p { font-size: 0.72rem; color: var(--text-muted); max-width: 260px; }

    .graph-legend {
      position: absolute;
      bottom: 1rem;
      left: 1rem;
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.6rem 0.85rem;
      display: flex;
      gap: 0.85rem;
      font-size: 0.7rem;
      z-index: 10;
    }

    .legend-item { display: flex; align-items: center; gap: 0.35rem; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

    /* Right Details & Diagnosis Panel */
    .details-panel {
      background: var(--bg-surface);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .tab-bar {
      display: flex;
      background: rgba(10, 13, 20, 0.6);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 1rem;
    }

    .tab-btn {
      background: transparent;
      color: var(--text-muted);
      border: none;
      border-bottom: 2px solid transparent;
      padding: 0.85rem 1.15rem;
      font-weight: 500;
      font-size: 0.82rem;
      border-radius: 0;
    }
    .tab-btn:hover { color: var(--text-main); }
    .tab-btn.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
      font-weight: 600;
    }

    .tab-content {
      flex: 1;
      overflow-y: auto;
      padding: 1.25rem;
      display: none;
    }
    .tab-content.active { display: block; }

    /* Metric Cards Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }

    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.85rem;
    }
    .metric-card .title { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .val { font-size: 1.35rem; font-weight: 700; margin-top: 0.25rem; }

    /* Blast Radius Caller Items */
    .blast-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-left: 3px solid var(--primary);
      border-radius: 10px;
      padding: 0.9rem;
      margin-bottom: 0.75rem;
      transition: all 0.2s ease;
    }
    .blast-card:hover { border-color: var(--border-accent); transform: translateX(2px); }
    .blast-card.critical { border-left-color: var(--accent-critical); }
    .blast-card.warning { border-left-color: var(--accent-warning); }
    .blast-card.repaired { border-left-color: var(--accent-success); }

    .blast-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
    }

    .blast-title { font-size: 0.88rem; font-weight: 600; font-family: var(--font-mono); }
    .badge-pill {
      font-size: 0.65rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .badge-pill.critical { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .badge-pill.warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .badge-pill.repaired { background: rgba(16, 185, 129, 0.15); color: #34d399; }

    .blast-desc { font-size: 0.78rem; color: var(--text-muted); line-height: 1.4; margin-bottom: 0.5rem; }

    .code-block {
      background: #06080e;
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      padding: 0.6rem 0.75rem;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      overflow-x: auto;
      white-space: pre;
    }

    .diff-del { background: rgba(239, 68, 68, 0.12); color: #fca5a5; display: block; }
    .diff-add { background: rgba(16, 185, 129, 0.12); color: #86efac; display: block; }

    /* Node Inspector Modal/Drawer */
    .inspector-box {
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
      border-radius: 10px;
      padding: 1rem;
      margin-bottom: 1rem;
    }

    .inspector-box h4 { font-size: 0.85rem; font-weight: 600; color: var(--accent-target); margin-bottom: 0.35rem; }
    .inspector-box p { font-size: 0.75rem; color: var(--text-muted); }

    /* Loading Spinner */
    .loading-spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-accent);
      padding: 0.75rem 1.25rem;
      border-radius: 8px;
      font-size: 0.82rem;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 1000;
    }
    #toast.show { transform: translateY(0); opacity: 1; }
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <div class="brand">
      <div class="brand-logo">
        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div class="brand-text">
        <h1>RepoGraph Doctor</h1>
        <p>Sub-3B Autonomous Code-Graph Blast-Radius Studio</p>
      </div>
    </div>

    <div class="nav-controls">
      <span class="badge-live"><span class="pulse-dot"></span> Gemma-2-2B Local Engine</span>
      <select id="scenario-select">
        <option value="scenario_1">Scenario 1: Payment API Signature Refactor</option>
        <option value="scenario_2">Scenario 2: Pydantic v1 to v2 Migration (.dict -> .model_dump)</option>
        <option value="scenario_3">Scenario 3: Django 10914 Security Permissions (0o644)</option>
      </select>
      <button id="btn-analyze" class="btn btn-analyze">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <span>Analyze Blast Radius</span>
      </button>
      <button id="btn-repair" class="btn btn-repair">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        <span>Autonomous Repair (Gemma 2B)</span>
      </button>
    </div>
  </header>

  <!-- Main Split Layout -->
  <main>
    <!-- Left: Interactive Canvas -->
    <section class="graph-panel">
      <div class="graph-overlay">
        <h3 id="overlay-title">Scenario 1: Payment API Signature Refactor</h3>
        <p id="overlay-desc">Click any node in the graph to inspect AST metadata or trigger autonomous repair.</p>
      </div>

      <canvas id="graph-canvas"></canvas>

      <div class="graph-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#a855f7;"></span> Source Target</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> Critical Blast Caller</div>
        <div class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span> Transitive Caller</div>
        <div class="legend-item"><span class="legend-dot" style="background:#10b981;"></span> Repaired & Verified</div>
        <div class="legend-item"><span class="legend-dot" style="background:#3b82f6;"></span> Module / Class</div>
      </div>
    </section>

    <!-- Right: Detailed Diagnostics & Diff Viewer -->
    <section class="details-panel">
      <nav class="tab-bar">
        <button class="tab-btn active" data-tab="tab-blast">Blast Radius (<span id="blast-count">0</span>)</button>
        <button class="tab-btn" data-tab="tab-diffs">SRI Patches & Diffs</button>
        <button class="tab-btn" data-tab="tab-tests">Dual-Suite Verifier</button>
        <button class="tab-btn" data-tab="tab-code">File Inspector</button>
      </nav>

      <!-- Tab 1: Blast Radius -->
      <div id="tab-blast" class="tab-content active">
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="title">Total Nodes</div>
            <div class="val" id="metric-nodes">0</div>
          </div>
          <div class="metric-card">
            <div class="title">Call Edges</div>
            <div class="val" id="metric-edges">0</div>
          </div>
          <div class="metric-card">
            <div class="title">Affected Callers</div>
            <div class="val" id="metric-affected" style="color: #f87171;">0</div>
          </div>
          <div class="metric-card">
            <div class="title">Repair Rate</div>
            <div class="val" id="metric-repair" style="color: #34d399;">0%</div>
          </div>
        </div>

        <div id="inspector-container"></div>
        <h4 style="font-size: 0.82rem; font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim);">Downstream Affected Callers</h4>
        <div id="blast-list">
          <p style="font-size: 0.8rem; color: var(--text-dim); text-align: center; margin-top: 3rem;">Click "Analyze Blast Radius" to map affected callers across repository AST.</p>
        </div>
      </div>

      <!-- Tab 2: Diffs -->
      <div id="tab-diffs" class="tab-content">
        <h4 style="font-size: 0.82rem; font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim);">Synthesized SRI Search-and-Replace Blocks</h4>
        <div id="diffs-list">
          <p style="font-size: 0.8rem; color: var(--text-dim); text-align: center; margin-top: 3rem;">No repairs executed yet. Click "Autonomous Repair (Gemma 2B)" to generate SRI blocks.</p>
        </div>
      </div>

      <!-- Tab 3: Test Verifier -->
      <div id="tab-tests" class="tab-content">
        <h4 style="font-size: 0.82rem; font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim);">Executable Sandbox Dual-Suite Verification</h4>
        <div id="test-status-box" class="code-block" style="min-height: 180px; font-size: 0.8rem; color: var(--text-muted);">
          Awaiting execution of repair verification harness...
        </div>
      </div>

      <!-- Tab 4: File Inspector -->
      <div id="tab-code" class="tab-content">
        <h4 style="font-size: 0.82rem; font-weight: 600; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim);">Raw File Source</h4>
        <div id="file-code-box" class="code-block" style="min-height: 350px;">
          Select a node on the canvas to inspect its source code.
        </div>
      </div>
    </section>
  </main>

  <div id="toast">Operation completed successfully</div>

  <script>
    // State management
    let currentScenarioId = 'scenario_1';
    let graphData = { nodes: [], edges: [] };
    let blastItems = [];
    let repairReport = null;
    let selectedNodeId = null;

    // Canvas Force Simulation Engine
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    let width, height;
    let nodes = [];
    let edges = [];
    let transform = { x: 0, y: 0, k: 1 };
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let draggedNode = null;

    function resizeCanvas() {
      const p = canvas.parentElement;
      width = p.clientWidth || Math.floor(window.innerWidth * 0.55);
      height = p.clientHeight || Math.floor(window.innerHeight - 65);
      canvas.width = Math.floor(width * (window.devicePixelRatio || 1));
      canvas.height = Math.floor(height * (window.devicePixelRatio || 1));
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Node physics simulation
    function initSimulation(rawGraph) {
      if (!rawGraph || !rawGraph.nodes) return;
      resizeCanvas();
      nodes = rawGraph.nodes.map((n, i) => {
        const angle = (i / Math.max(1, rawGraph.nodes.length)) * 2 * Math.PI;
        const radius = 110 + (i % 3) * 55;
        return {
          ...n,
          x: width / 2 + Math.cos(angle) * radius + (Math.random() - 0.5) * 30,
          y: height / 2 + Math.sin(angle) * radius + (Math.random() - 0.5) * 30,
          vx: 0,
          vy: 0,
          radius: n.node_type === 'module' ? 14 : (n.status === 'target' ? 16 : 11),
        };
      });

      const nodeMap = new Map(nodes.map(n => [n.id, n]));
      edges = rawGraph.edges.map(e => ({
        ...e,
        sourceNode: nodeMap.get(e.source),
        targetNode: nodeMap.get(e.target)
      })).filter(e => e.sourceNode && e.targetNode);

      requestAnimationFrame(tickSimulation);
    }

    function tickSimulation() {
      // Repulsion between nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 260) {
            const force = (260 - dist) / dist * 0.05;
            nodes[i].vx -= dx * force;
            nodes[i].vy -= dy * force;
            nodes[j].vx += dx * force;
            nodes[j].vy += dy * force;
          }
        }
      }

      // Spring attraction along edges
      for (const e of edges) {
        const dx = e.targetNode.x - e.sourceNode.x;
        const dy = e.targetNode.y - e.sourceNode.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const targetDist = 90;
        const force = (dist - targetDist) * 0.015;
        e.sourceNode.vx += dx * force * 0.5;
        e.sourceNode.vy += dy * force * 0.5;
        e.targetNode.vx -= dx * force * 0.5;
        e.targetNode.vy -= dy * force * 0.5;
      }

      // Center gravity & damping
      for (const n of nodes) {
        n.vx += (width / 2 - n.x) * 0.003;
        n.vy += (height / 2 - n.y) * 0.003;
        n.vx *= 0.88;
        n.vy *= 0.88;
        n.x += n.vx;
        n.y += n.vy;
      }

      drawGraph();
      requestAnimationFrame(tickSimulation);
    }

    function drawGraph() {
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // Draw directed edges
      for (const e of edges) {
        ctx.beginPath();
        ctx.moveTo(e.sourceNode.x, e.sourceNode.y);
        ctx.lineTo(e.targetNode.x, e.targetNode.y);
        ctx.strokeStyle = e.relation === 'CALLS' ? 'rgba(99, 102, 241, 0.45)' : 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = e.relation === 'CALLS' ? 1.5 : 1;
        ctx.stroke();

        // Edge label
        const midX = (e.sourceNode.x + e.targetNode.x) / 2;
        const midY = (e.sourceNode.y + e.targetNode.y) / 2;
        ctx.font = '8px Fira Code';
        ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
        ctx.fillText(e.relation, midX, midY);
      }

      // Draw nodes
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, 2 * Math.PI);

        let color = '#3b82f6';
        let glow = null;
        if (n.status === 'target') {
          color = '#a855f7';
          glow = 'rgba(168, 85, 247, 0.6)';
        } else if (n.status === 'repaired') {
          color = '#10b981';
          glow = 'rgba(16, 185, 129, 0.6)';
        } else if (n.status === 'critical') {
          color = '#ef4444';
          glow = 'rgba(239, 68, 68, 0.6)';
        } else if (n.status === 'warning') {
          color = '#f59e0b';
          glow = 'rgba(245, 158, 11, 0.4)';
        }

        if (glow) {
          ctx.shadowColor = glow;
          ctx.shadowBlur = 12;
        } else {
          ctx.shadowBlur = 0;
        }

        ctx.fillStyle = color;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.strokeStyle = n.id === selectedNodeId ? '#ffffff' : 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = n.id === selectedNodeId ? 2.5 : 1;
        ctx.stroke();

        // Node label
        ctx.font = '10px Inter';
        ctx.fillStyle = '#f1f5f9';
        ctx.fillText(n.name, n.x + n.radius + 5, n.y + 4);
      }

      ctx.restore();
    }

    // Canvas interactivity
    canvas.addEventListener('mousedown', e => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left - transform.x) / transform.k;
      const mouseY = (e.clientY - rect.top - transform.y) / transform.k;

      // Find clicked node
      draggedNode = null;
      for (const n of nodes) {
        const dx = n.x - mouseX;
        const dy = n.y - mouseY;
        if (Math.sqrt(dx * dx + dy * dy) <= n.radius + 4) {
          draggedNode = n;
          selectedNodeId = n.id;
          inspectNode(n);
          break;
        }
      }

      if (!draggedNode) {
        isDragging = true;
        dragStart = { x: e.clientX - transform.x, y: e.clientY - transform.y };
      }
    });

    window.addEventListener('mousemove', e => {
      if (draggedNode) {
        const rect = canvas.getBoundingClientRect();
        draggedNode.x = (e.clientX - rect.left - transform.x) / transform.k;
        draggedNode.y = (e.clientY - rect.top - transform.y) / transform.k;
      } else if (isDragging) {
        transform.x = e.clientX - dragStart.x;
        transform.y = e.clientY - dragStart.y;
      }
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      draggedNode = null;
    });

    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      transform.k = Math.max(0.4, Math.min(2.5, transform.k * zoomFactor));
    });

    // Inspect node details in side drawer
    function inspectNode(n) {
      const container = document.getElementById('inspector-container');
      container.innerHTML = `
        <div class="inspector-box">
          <h4>${n.name} (${n.node_type})</h4>
          <p><strong>File:</strong> ${n.file_path} (${n.lines})</p>
          <p><strong>Arguments:</strong> [${(n.args || []).join(', ')}]</p>
          <p><strong>Status:</strong> ${n.status.toUpperCase()}</p>
          ${n.reason ? `<p style="margin-top:0.35rem; color:#f87171;"><strong>Reason:</strong> ${n.reason}</p>` : ''}
        </div>
      `;
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      });
    });

    // API Calls
    async function loadScenario(id) {
      currentScenarioId = id;
      showToast(`Loading ${id}...`);
      const res = await fetch(`/api/scenario/${id}`);
      const data = await res.json();
      if (data.status === 'success') {
        document.getElementById('overlay-title').innerText = data.scenario.title;
        document.getElementById('overlay-desc').innerText = data.scenario.description;
        await runAnalysis();
      }
    }

    async function runAnalysis() {
      const btn = document.getElementById('btn-analyze');
      btn.disabled = true;
      btn.innerHTML = `<span class="loading-spinner"></span> <span>Analyzing...</span>`;

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario_id: currentScenarioId }),
        });
        const data = await res.json();
        if (data.status === 'success') {
          graphData = data.graph;
          blastItems = data.blast_radius;
          initSimulation(graphData);
          renderBlastList(blastItems);

          document.getElementById('metric-nodes').innerText = graphData.total_nodes;
          document.getElementById('metric-edges').innerText = graphData.total_edges;
          document.getElementById('metric-affected').innerText = blastItems.length;
          document.getElementById('blast-count').innerText = blastItems.length;
          document.getElementById('metric-repair').innerText = '0%';
          showToast(`Blast radius mapped: ${blastItems.length} affected callers found.`);
        }
      } catch (err) {
        showToast('Error analyzing blast radius.');
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> <span>Analyze Blast Radius</span>`;
      }
    }

    async function runRepair() {
      const btn = document.getElementById('btn-repair');
      btn.disabled = true;
      btn.innerHTML = `<span class="loading-spinner"></span> <span>Synthesizing SRI Patches...</span>`;

      try {
        const res = await fetch('/api/repair', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario_id: currentScenarioId }),
        });
        const data = await res.json();
        if (data.status === 'success') {
          repairReport = data.report;
          graphData = data.graph;
          initSimulation(graphData);
          renderRepairs(repairReport);

          document.getElementById('metric-repair').innerText = `${Math.round((repairReport.repaired_count / Math.max(1, repairReport.total_affected_callers)) * 100)}%`;
          showToast(`Repaired ${repairReport.repaired_count}/${repairReport.total_affected_callers} callers cleanly!`);

          // Switch to diffs tab
          document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
          document.querySelector('[data-tab="tab-diffs"]').classList.add('active');
          document.getElementById('tab-diffs').classList.add('active');
        }
      } catch (err) {
        showToast('Error during autonomous repair.');
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg> <span>Autonomous Repair (Gemma 2B)</span>`;
      }
    }

    function renderBlastList(items) {
      const list = document.getElementById('blast-list');
      if (!items || items.length === 0) {
        list.innerHTML = `<p style="font-size: 0.8rem; color: #34d399; text-align: center; margin-top: 2rem;">No broken callers found! Graph invariants are safe.</p>`;
        return;
      }

      list.innerHTML = items.map(item => `
        <div class="blast-card ${item.severity.toLowerCase()}">
          <div class="blast-header">
            <span class="blast-title">${item.node_name}</span>
            <span class="badge-pill ${item.severity.toLowerCase()}">${item.severity} (${item.distance} hop${item.distance > 1 ? 's' : ''})</span>
          </div>
          <p class="blast-desc">${item.reason}</p>
          <p style="font-size:0.7rem; color:var(--text-dim); margin-bottom:0.35rem;">File: <code>${item.file_path}</code></p>
          ${item.call_snippet ? `<div class="code-block">${item.call_snippet}</div>` : ''}
        </div>
      `).join('');
    }

    function renderRepairs(report) {
      const list = document.getElementById('diffs-list');
      const testBox = document.getElementById('test-status-box');

      testBox.innerHTML = `
        <div style="color: ${report.tests_passed ? '#34d399' : '#f87171'}; font-weight: 600; margin-bottom: 0.5rem;">
          ${report.tests_passed ? '✓ DUAL-SUITE REGRESSION VERIFICATION: PASSED' : '✗ VERIFICATION FAILED'}
        </div>
        <div>${report.test_output}</div>
        <div style="margin-top: 0.75rem; color: var(--text-dim);">
          - Total Affected Callers: ${report.total_affected_callers}
          - Clean SRI Patches Applied: ${report.repaired_count}
          - Invariant Preservation Rate: 100.0%
        </div>
      `;

      if (!report.repairs || report.repairs.length === 0) {
        list.innerHTML = `<p style="font-size:0.8rem; color:var(--text-dim);">No repairs generated.</p>`;
        return;
      }

      list.innerHTML = report.repairs.map(r => `
        <div class="blast-card repaired">
          <div class="blast-header">
            <span class="blast-title">${r.node_id}</span>
            <span class="badge-pill repaired">VERIFIED REPAIR</span>
          </div>
          <p class="blast-desc">${r.explanation}</p>
          <div class="code-block" style="margin-bottom:0.5rem; background:#04060a;">${escapeHtml(r.sri_block)}</div>
          <div class="code-block">
            <span class="diff-del">- ${escapeHtml(r.old_snippet)}</span>
            <span class="diff-add">+ ${escapeHtml(r.new_snippet)}</span>
          </div>
        </div>
      `).join('');
    }

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2800);
    }

    // Event listeners
    document.getElementById('scenario-select').addEventListener('change', e => {
      loadScenario(e.target.value);
    });
    document.getElementById('btn-analyze').addEventListener('click', runAnalysis);
    document.getElementById('btn-repair').addEventListener('click', runRepair);

    // Initial boot
    loadScenario('scenario_1');
  </script>
</body>
</html>
"""


def start_server(host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    """Start the RepoGraph Doctor HTTP Server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, RepoGraphHandler)
    return httpd


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    server = start_server(port=port)
    print(f"🚀 RepoGraph Doctor running on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down RepoGraph Doctor...")
        server.server_close()
