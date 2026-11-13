---
name: repo_navigation
description: Best practices and heuristics for exploring code graphs, identifying blast radius, and writing minimal edits with edit_file.
---

# Repository Navigation & Code Graph Skill

This skill provides operational heuristics for resolving repository bugs using pre-computed embeddings and call graphs.

## Recommended Workflow

### 1. Code Graph Queries
- When an issue refers to a class or method, do not read files sequentially. Call `search_similar_code(query="<symbol_name>")`.
- To see all places where a function is invoked, call `get_code_neighbors(node="<symbol>", edge_type="CALLS")`.
- If a change might break callers, call `get_code_neighbors` to verify if any caller expects the old return type or argument signature.

### 2. Precise Editing with `edit_file`
- Before modifying a file, call `read_file` with a 30-50 line window around the target lines.
- Copy the exact target lines into `old_string`. Include enough surrounding lines to make the match unique.
- Formulate `new_string` preserving existing indentation (spaces vs tabs), imports, and type hints.
- Never delete error handling or docstrings unless explicitly required by the issue.

### 3. Test Verification
- Run tests using `run_command(command="pytest <path_to_test>")`.
- If tests fail, inspect the traceback before attempting another edit.
