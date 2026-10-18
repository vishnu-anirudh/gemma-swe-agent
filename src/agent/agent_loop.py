"""Interactive Multi-Turn Agent Loop for Repository-Level Bug Fixing."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.agent.tools import RepoTools
from src.data.masking import TrajectoryStep
from src.data.sri_formatter import SRIBlock, SRIFormatter
from src.evaluation.patch_applicator import PatchApplicator
from rich.console import Console

console = Console()


@dataclass
class AgentSession:
    """Records the trajectory, interaction history, and final patch of an agent run."""

    instance_id: str
    turns: int
    tool_calls_made: int
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    final_patch_raw: Optional[str] = None
    parsed_sri_blocks: List[SRIBlock] = field(default_factory=list)
    files_viewed: List[str] = field(default_factory=list)
    resolved: bool = False
    error: Optional[str] = None


class MultiTurnAgent:
    """Orchestrates multi-turn interactive codebase exploration and patch generation."""

    AGENT_SYSTEM_PROMPT = (
        "You are an expert autonomous software engineering agent tasked with resolving a repository bug.\n"
        "You can inspect files, search for symbols, and run tests before submitting your fix.\n\n"
        "Available Tools:\n"
        "1. search_code: Search symbol or text across the repository\n"
        "   Example: ```tool_call\nsearch_code query=\"function_name\"\n```\n"
        "   Or: <tool_call name=\"search_code\" query=\"function_name\" />\n\n"
        "2. view_file: View lines of a file\n"
        "   Example: ```tool_call\nview_file path=\"path/to/file.py\" start_line=\"1\" end_line=\"50\"\n```\n"
        "   Or: <tool_call name=\"view_file\" path=\"path/to/file.py\" start_line=\"1\" end_line=\"50\" />\n\n"
        "3. list_dir: List directory contents\n"
        "   Example: ```tool_call\nlist_dir path=\"src/\"\n```\n\n"
        "4. run_test: Execute test suite\n"
        "   Example: ```tool_call\nrun_test command=\"pytest tests/test_calc.py\"\n```\n\n"
        "When you have identified the fix, submit it in Search-and-Replace Infilling (SRI) format:\n"
        "<submit_patch>\n"
        "<<<<<<< SEARCH: relative/path/to/file.py\n"
        "<exact original lines to search for copied verbatim from view_file>\n"
        "=======\n"
        "<replacement code>\n"
        ">>>>>>> REPLACE\n"
        "</submit_patch>\n\n"
        "CRITICAL RULES for SEARCH block:\n"
        "1. The SEARCH block MUST contain ONLY code that currently exists in the file, copied verbatim from view_file.\n"
        "2. Do NOT add new comments, placeholder comments, or new code inside SEARCH.\n"
        "3. Put ALL additions, new comments, and fixes in the REPLACE block only.\n"
        "4. Search blocks MUST match the actual file contents exactly."
    )

    PATCH_PATTERN = re.compile(r"<submit_patch>(.*?)</submit_patch>", re.DOTALL)

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tokenizer: Optional[Any] = None,
        max_turns: int = 5,
    ):
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer
        self.max_turns = max_turns

    @classmethod
    def parse_tool_call(cls, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract tool call across XML tags, Markdown blocks, and CLI styles."""
        # 1. XML style: <tool_call name="..." ... />
        xml_match = re.search(r'<tool_call\s+name="([^"]+)"([^>]*)/?>', text)
        if xml_match:
            tool_name = xml_match.group(1).strip()
            attrs_str = xml_match.group(2)
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))
            return tool_name, attrs

        def _extract_attrs(tool_str: str, args_str: str) -> Dict[str, Any]:
            clean_attrs = {}
            for m in re.finditer(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s\)]+))', args_str):
                k = m.group(1)
                v = m.group(2) or m.group(3) or m.group(4) or ""
                clean_attrs[k] = v
            if tool_str == "search_code" and "name" in clean_attrs and "query" not in clean_attrs:
                clean_attrs["query"] = clean_attrs["name"]
            return clean_attrs

        # 2. Markdown code blocks (```tool_call, ```python, ```sh, etc.)
        for block_match in re.finditer(r'```(?:tool_call|python|bash|sh)?\s*\n?(.*?)\n?```', text, re.DOTALL):
            block_content = block_match.group(1).strip()
            m = re.search(r'\b(search_code|view_file|list_dir|run_test)\b\s*\(?([^)\n]*)\)?', block_content)
            if m:
                t_name = m.group(1).strip()
                attrs = _extract_attrs(t_name, m.group(2))
                if attrs:
                    return t_name, attrs

        # 3. Line-anchored tool calls (starting at beginning of a line)
        line_match = re.search(
            r'^\s*(?:`{1,3})?\s*(search_code|view_file|list_dir|run_test)\b\s*\(?([^)\n]*)\)?',
            text,
            re.MULTILINE,
        )
        if line_match:
            t_name = line_match.group(1).strip()
            attrs = _extract_attrs(t_name, line_match.group(2))
            if attrs:
                return t_name, attrs

        # 4. Fallback search across free text, only if attributes are present
        for m in re.finditer(r'\b(search_code|view_file|list_dir|run_test)\b\s*\(?([^)\n]*)\)?', text):
            t_name = m.group(1).strip()
            attrs = _extract_attrs(t_name, m.group(2))
            if attrs:
                return t_name, attrs

        return None

    @staticmethod
    def is_file_viewed(target_path: str, files_viewed: Set[str]) -> bool:
        """Verify whether target_path has been inspected with view_file."""
        target_norm = target_path.strip().replace("\\", "/").lstrip("./")
        if not target_norm:
            return False
        for v in files_viewed:
            v_norm = v.strip().replace("\\", "/").lstrip("./")
            if not v_norm:
                continue
            if (
                target_norm == v_norm
                or target_norm.endswith("/" + v_norm)
                or v_norm.endswith("/" + target_norm)
            ):
                return True
            t_parts = target_norm.split("/")
            v_parts = v_norm.split("/")
            if len(t_parts) >= 2 and len(v_parts) >= 2 and t_parts[-2:] == v_parts[-2:]:
                return True
        return False

    def _prune_messages(self, messages: List[Dict[str, str]], max_older_obs_lines: int = 12) -> List[Dict[str, str]]:
        """Keep initial issue prompt and recent turns intact, while summarizing older observations."""
        if len(messages) <= 6:
            return messages

        pruned = [messages[0]]
        cutoff = len(messages) - 4
        for idx in range(1, len(messages)):
            msg = messages[idx]
            if (
                idx < cutoff
                and msg["role"] == "user"
                and "Tool Observation for" in msg["content"]
                and "=== File:" not in msg["content"]
                and "=== Auto-Inspected File:" not in msg["content"]
            ):
                lines = msg["content"].splitlines()
                if len(lines) > max_older_obs_lines:
                    compressed = "\n".join(lines[:max_older_obs_lines]) + "\n... [earlier observation truncated to preserve context budget]"
                    pruned.append({"role": "user", "content": compressed})
                    continue
            pruned.append(msg)
        return pruned

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using tokenizer chat template if available, else plain text."""
        pruned_msgs = self._prune_messages(messages)
        # Ensure strict alternation for chat templates (e.g. Gemma requires alternating user/model)
        normalized: List[Dict[str, str]] = []
        for msg in pruned_msgs:
            if normalized and normalized[-1]["role"] == msg["role"]:
                normalized[-1] = {
                    "role": msg["role"],
                    "content": normalized[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                normalized.append(dict(msg))

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Chat template failed ({e}), falling back to dialogue format.[/yellow]")

        # Fallback to standard dialogue format
        rendered = []
        for msg in normalized:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                rendered.append(f"User:\n{content}")
            else:
                rendered.append(f"Assistant:\n{content}")
        rendered.append("Assistant:\n")
        return "\n\n".join(rendered)

    def run_session(
        self,
        instance_id: str,
        problem_statement: str,
        repo_dir: str,
        test_command: Optional[str] = None,
        test_timeout: int = 30,
    ) -> AgentSession:
        """Execute a multi-turn exploration and bug-fixing session."""
        session = AgentSession(instance_id=instance_id, turns=0, tool_calls_made=0)
        viewed_files_set: Set[str] = set()
        seen_tool_calls: Set[Tuple[str, Tuple[Tuple[str, Any], ...]]] = set()
        last_inspected_snippet: str = ""
        traceback_targets: Dict[str, int] = {}

        initial_test_failure = ""
        if test_command:
            try:
                proc = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=test_timeout,
                )
                if proc.returncode != 0:
                    full_tb = proc.stdout + "\n" + proc.stderr
                    err_lines = [l for l in full_tb.splitlines() if l.strip()]
                    err_snip = "\n".join(err_lines[-15:])
                    initial_test_failure = f"\n=== Reproducing Test Suite Output ===\n{err_snip}\n"
                    console.print(f"  [dim cyan]-> Seeded with reproduction test failure: {err_lines[-1] if err_lines else ''}[/dim cyan]")

                    # Extract in-repo non-test source frames from traceback
                    repo_abs = str(Path(repo_dir).resolve())
                    for m_tb in re.finditer(r'File "([^"]+\.py)", line (\d+)', full_tb):
                        raw_p, tb_l = m_tb.group(1), int(m_tb.group(2))
                        try:
                            rel_p = str(Path(raw_p).resolve().relative_to(repo_abs))
                        except Exception:
                            rel_p = raw_p.lstrip("./")
                        if (
                            not rel_p.startswith("tests/")
                            and "/tests/" not in rel_p
                            and (Path(repo_dir) / rel_p).is_file()
                            and rel_p not in traceback_targets
                        ):
                            traceback_targets[rel_p] = tb_l
                            tb_start = max(1, tb_l - 4)
                            tb_end = tb_l + 10
                            try:
                                file_lines = (Path(repo_dir) / rel_p).read_text(encoding="utf-8", errors="replace").splitlines()
                                for scan_l in range(tb_l, max(1, tb_l - 12) - 1, -1):
                                    if scan_l - 1 < len(file_lines) and file_lines[scan_l - 1].lstrip().startswith("def "):
                                        tb_start = scan_l
                                        break
                                for end_l_scan in range(tb_start + 1, min(len(file_lines), tb_l + 15) + 1):
                                    if file_lines[end_l_scan - 1].lstrip().startswith(("def ", "class ")):
                                        tb_end = end_l_scan - 1
                                        break
                            except Exception:
                                pass
                            last_inspected_snippet = RepoTools.view_file(repo_dir, rel_p, tb_start, tb_end)
                            viewed_files_set.add(rel_p)
                            session.files_viewed.append(rel_p)
                            break
            except Exception:
                pass

        initial_user_msg = (
            f"{self.AGENT_SYSTEM_PROMPT}\n\n"
            f"=== Issue Report for {instance_id} ===\n"
            f"{problem_statement}\n"
            f"{initial_test_failure}\n"
            f"Search Strategy Tip: When searching for symbols, use short single-word class, function, or setting names "
            f"(e.g. search_code query='ClassName' or query='SETTING_NAME') rather than long sentences or phrases.\n"
            f"Please begin by searching the codebase or viewing relevant files using a tool call."
        )

        messages = [{"role": "user", "content": initial_user_msg}]

        for turn in range(1, self.max_turns + 1):
            session.turns = turn
            prompt_str = self._format_conversation(messages)
            model_response = self.generate_fn(prompt_str).strip()

            # 1. Parse potential tool call and patch
            patch_match = self.PATCH_PATTERN.search(model_response)
            parsed_tool = self.parse_tool_call(model_response)
            direct_sri_blocks = SRIFormatter.parse(model_response) if not patch_match else []

            # If tool call is present and there is NO explicit <submit_patch> tag, prioritize tool execution
            if parsed_tool is not None and not patch_match:
                pass
            elif patch_match or direct_sri_blocks:
                if patch_match:
                    patch_text = patch_match.group(1).strip()
                    sri_blocks = SRIFormatter.parse(patch_text)
                    is_raw = False
                else:
                    patch_text = model_response
                    sri_blocks = direct_sri_blocks
                    is_raw = True

                if patch_match and not sri_blocks and turn < self.max_turns:
                    console.print(
                        f"  [yellow]Turn {turn}/{self.max_turns}:[/yellow] Malformed patch syntax inside <submit_patch>. Rejecting."
                    )
                    messages.append({"role": "model", "content": model_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "SYSTEM DIRECTIVE: Could not parse any valid SEARCH/REPLACE blocks from your <submit_patch> tag.\n"
                                "Please ensure your patch strictly adheres to this structure:\n"
                                "<submit_patch>\n"
                                "<<<<<<< SEARCH: relative/path/to/file.py\n"
                                "<exact original lines to search for copied verbatim from view_file>\n"
                                "=======\n"
                                "<replacement code>\n"
                                ">>>>>>> REPLACE\n"
                                "</submit_patch>"
                            ),
                        }
                    )
                    continue

                if sri_blocks:
                    # Premature Patch Guard: Enforce inspecting target file first before submitting patch
                    unviewed = [
                        b.file_path
                        for b in sri_blocks
                        if not self.is_file_viewed(b.file_path, viewed_files_set)
                    ]
                    if unviewed and turn < self.max_turns:
                        target_f = unviewed[0]
                        fname = Path(target_f).name
                        suggested_start = 1
                        suggested_end = 60
                        for step in reversed(session.trajectory_steps):
                            m = re.search(re.escape(fname) + r":(\d+):", step.observation_text)
                            if m:
                                line_num = int(m.group(1))
                                suggested_start = max(1, line_num - 15)
                                suggested_end = line_num + 35
                                break

                        console.print(
                            f"  [yellow]Turn {turn}/{self.max_turns}:[/yellow] Premature patch rejected for uninspected file: '{target_f}'. Enforcing view_file."
                        )
                        session.trajectory_steps.append(
                            TrajectoryStep(
                                action_text=model_response,
                                observation_text=f"Guard rejected premature patch on uninspected file: {target_f}",
                                is_error=True,
                            )
                        )
                        messages.append({"role": "model", "content": model_response})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"SYSTEM DIRECTIVE: Premature patch rejected! You attempted to modify '{target_f}' "
                                    f"without inspecting it first using view_file.\n"
                                    f"To guarantee that your SEARCH block matches the file verbatim, you MUST inspect the target lines first:\n"
                                    f"```tool_call\nview_file path=\"{target_f}\" start_line=\"{suggested_start}\" end_line=\"{suggested_end}\"\n```\n"
                                    f"Please view the source file now."
                                ),
                            }
                        )
                        continue

                    # No-Op Patch Guard: Enforce actual modifications
                    is_noop = all(b.search_content.strip() == b.replace_content.strip() for b in sri_blocks)
                    if is_noop and turn < self.max_turns:
                        console.print(f"  [yellow]Turn {turn}/{self.max_turns}:[/yellow] No-op patch rejected! SEARCH and REPLACE blocks are identical.")
                        session.trajectory_steps.append(
                            TrajectoryStep(
                                action_text=model_response,
                                observation_text="Guard rejected no-op patch where SEARCH == REPLACE.",
                                is_error=True,
                            )
                        )
                        messages.append({"role": "model", "content": model_response})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "SYSTEM DIRECTIVE: No-op patch rejected! Your REPLACE block contains the exact same code as your SEARCH block.\n"
                                    "A patch must contain code modifications to fix the bug.\n"
                                    "Please write the actual replacement code inside the REPLACE block."
                                ),
                            }
                        )
                        continue

                    # If test_command is provided, perform test-driven verification & feedback loop
                    if test_command:
                        console.print(f"  [cyan]Turn {turn}/{self.max_turns}:[/cyan] Verifying candidate patch against test suite...")
                        apply_res = PatchApplicator.extract_and_apply(patch_text, repo_dir)
                        if not apply_res.success:
                            if turn < self.max_turns:
                                console.print(f"  [yellow]Turn {turn}/{self.max_turns}:[/yellow] Patch application failed: {apply_res.error_message}")
                                messages.append({"role": "model", "content": model_response})
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            f"SYSTEM DIRECTIVE: Patch application failed: {apply_res.error_message}\n"
                                            f"Your SEARCH block did not match the file. Please view the file using view_file and ensure exact verbatim code in SEARCH."
                                        ),
                                    }
                                )
                                continue
                        else:
                            try:
                                proc = subprocess.run(
                                    test_command,
                                    shell=True,
                                    cwd=repo_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=test_timeout,
                                )
                                if proc.returncode == 0:
                                    console.print(f"  [bold green]Turn {turn}/{self.max_turns}: Verification SUCCESS! Test suite passed.[/bold green]")
                                    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                                    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                                    session.final_patch_raw = patch_text
                                    session.parsed_sri_blocks = sri_blocks
                                    session.resolved = True
                                    action_label = "submit_patch" if not is_raw else "submit_patch_raw"
                                    session.trajectory_steps.append(
                                        TrajectoryStep(
                                            action_text=action_label,
                                            observation_text="Patch verified: tests passed.",
                                            is_error=False,
                                        )
                                    )
                                    break
                                else:
                                    # Candidate patch failed test suite: revert sandbox to HEAD and provide error feedback
                                    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                                    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                                    err_lines = [l for l in (proc.stdout + "\n" + proc.stderr).splitlines() if l.strip()]
                                    err_snippet = "\n".join(err_lines[-12:])
                                    console.print(f"  [yellow]Turn {turn}/{self.max_turns}: Test failed (exit code {proc.returncode}). Reverting sandbox and prompting refinement...[/yellow]")
                                    for b in sri_blocks:
                                        console.print(f"    [dim yellow]-> Candidate block: file={b.file_path} | search={repr(b.search_content[:50])} | replace={repr(b.replace_content[:50])}[/dim yellow]")
                                    if err_lines:
                                        console.print(f"    [dim red]-> Error: {err_lines[-1]}[/dim red]")

                                    if turn < self.max_turns:
                                        messages.append({"role": "model", "content": model_response})
                                        messages.append(
                                            {
                                                "role": "user",
                                                "content": (
                                                    f"Test Verification FAILED (exit code {proc.returncode}):\n{err_snippet}\n\n"
                                                    f"SYSTEM DIRECTIVE: Your candidate patch was applied, but the test suite failed with the output above.\n"
                                                    f"The repository sandbox has been reverted to clean state.\n"
                                                    f"Please analyze the failure (e.g. search for configuration settings or the failing symbols) and submit a corrected fix."
                                                ),
                                            }
                                        )
                                        continue
                            except subprocess.TimeoutExpired:
                                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                                if turn < self.max_turns:
                                    console.print(f"  [yellow]Turn {turn}/{self.max_turns}: Test verification timed out after {test_timeout}s.[/yellow]")
                                    messages.append({"role": "model", "content": model_response})
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": f"Test Verification TIMED OUT after {test_timeout}s. Please submit an alternative or simpler fix.",
                                        }
                                    )
                                    continue

                    # Patch is validated and accepted
                    session.final_patch_raw = patch_text
                    session.parsed_sri_blocks = sri_blocks
                    action_label = "submit_patch" if not is_raw else "submit_patch_raw"
                    obs_label = "Patch submitted." if not is_raw else "Direct SRI patch detected."
                    session.trajectory_steps.append(
                        TrajectoryStep(
                            action_text=action_label,
                            observation_text=obs_label,
                            is_error=False,
                        )
                    )
                    console.print(
                        f"  [bold green]Turn {turn}/{self.max_turns}:[/bold green] Patch accepted ({len(session.parsed_sri_blocks)} SRI blocks)."
                    )
                    for b in session.parsed_sri_blocks:
                        console.print(
                            f"    [dim cyan]-> File: {b.file_path} | Search: {repr(b.search_content[:80])} | Replace: {repr(b.replace_content[:80])}[/dim cyan]"
                        )
                    break

            # 2. Check for tool calls using multi-format parser
            parsed = parsed_tool
            if parsed:
                tool_name, attrs = parsed
                path = attrs.get("path", ".")
                query = attrs.get("query", attrs.get("name", ""))
                command = attrs.get("command", "")
                start_l = int(attrs.get("start_line", 1))
                end_l = int(attrs.get("end_line", 100))

                # Auto-route search_code with line ranges or python paths to view_file
                if tool_name == "search_code" and ("start_line" in attrs or path.endswith(".py")):
                    tool_name = "view_file"

                session.tool_calls_made += 1

                # Check for duplicate tool calls
                tool_sig = (tool_name, tuple(sorted(attrs.items())))
                is_duplicate = tool_sig in seen_tool_calls or (
                    len(session.trajectory_steps) > 0
                    and session.trajectory_steps[-1].action_text.strip() == model_response.strip()
                )
                seen_tool_calls.add(tool_sig)

                if (is_duplicate or (tool_name == "search_code" and turn > 1)) and session.files_viewed:
                    console.print(
                        f"  [cyan]Turn {turn}/{self.max_turns}:[/cyan] Secondary {tool_name} after code inspection; transitioning to patch synthesis..."
                    )
                    break

                if is_duplicate:
                    observation = (
                        f"Error: Duplicate tool call blocked! You already executed '{tool_name}' with these exact arguments in the previous turn.\n"
                        f"Do NOT repeat this call. You must either:\n"
                        f"1. Use view_file path=\"...\" start_line=\"...\" end_line=\"...\" to inspect code files identified above.\n"
                        f"2. Search for a different keyword or class/function symbol.\n"
                        f"3. Submit your patch wrapped inside <submit_patch>."
                    )
                elif tool_name == "view_file" and not attrs.get("path"):
                    observation = "Error: 'path' argument is required for view_file. Usage: ```tool_call\nview_file path=\"path/to/file.py\" start_line=\"1\" end_line=\"60\"\n```"
                elif tool_name == "search_code" and not attrs.get("query", attrs.get("name")):
                    observation = "Error: 'query' argument is required for search_code. Usage: ```tool_call\nsearch_code query=\"symbol_name\"\n```"
                elif tool_name == "view_file":
                    viewed_files_set.add(path)
                    session.files_viewed.append(path)
                    observation = RepoTools.view_file(repo_dir, path, start_l, end_l)
                    if "Error:" not in observation:
                        last_inspected_snippet = observation
                elif tool_name == "list_dir":
                    observation = RepoTools.list_dir(repo_dir, path)
                elif tool_name == "search_code":
                    observation = RepoTools.search_code(repo_dir, query)
                elif tool_name == "run_test":
                    observation = RepoTools.run_test(repo_dir, command)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'."

                is_err = "Error:" in observation or "FAILED" in observation

                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text=model_response,
                        observation_text=observation,
                        is_error=is_err,
                    )
                )

                first_obs = observation.splitlines()[0] if observation.splitlines() else observation
                console.print(f"  [cyan]Turn {turn}/{self.max_turns}:[/cyan] Tool Call -> [bold]{tool_name}[/bold]({attrs})")
                console.print(f"    [dim]-> Observation: {first_obs[:85]}[/dim]")

                hint = ""
                if tool_name == "search_code" and not is_duplicate:
                    m_match = re.search(r'^([a-zA-Z0-9_\-\./]+\.py):(\d+):(.*)$', observation, re.MULTILINE)
                    if m_match:
                        matched_path = m_match.group(1)
                        matched_line = int(m_match.group(2))
                        matched_code = m_match.group(3).strip()
                        if matched_path in traceback_targets:
                            tb_line = traceback_targets[matched_path]
                            suggested_start = max(1, tb_line - 4)
                            suggested_end = tb_line + 10
                            try:
                                file_lines = (Path(repo_dir) / matched_path).read_text(encoding="utf-8", errors="replace").splitlines()
                                for scan_l in range(tb_line, max(1, tb_line - 12) - 1, -1):
                                    if scan_l - 1 < len(file_lines) and file_lines[scan_l - 1].lstrip().startswith("def "):
                                        suggested_start = scan_l
                                        break
                                for end_l_scan in range(suggested_start + 1, min(len(file_lines), tb_line + 15) + 1):
                                    if file_lines[end_l_scan - 1].lstrip().startswith(("def ", "class ")):
                                        suggested_end = end_l_scan - 1
                                        break
                            except Exception:
                                pass
                        elif re.match(r"^[A-Z_][A-Z0-9_]*\s*=", matched_code):
                            suggested_start = matched_line
                            suggested_end = matched_line + 15
                        else:
                            suggested_start = max(1, matched_line - 5)
                            suggested_end = matched_line + 25
                        # Forced View After Search: auto-execute view_file so model always gets
                        # verbatim source before synthesising any patch
                        if turn < self.max_turns:
                            view_obs = RepoTools.view_file(repo_dir, matched_path, suggested_start, suggested_end)
                            viewed_files_set.add(matched_path)
                            session.files_viewed.append(matched_path)
                            last_inspected_snippet = view_obs
                            console.print(f"  [cyan]Turn {turn}/{self.max_turns}:[/cyan] [dim](auto)[/dim] view_file({matched_path} {suggested_start}:{suggested_end})")
                            top_match_line = observation.splitlines()[0] if observation.splitlines() else observation
                            observation = (
                                f"Search Match:\n{top_match_line}\n\n"
                                f"=== Auto-Inspected File: {matched_path} (lines {suggested_start}-{suggested_end}) ===\n"
                                f"{view_obs}"
                            )
                            hint = (
                                "\n[Directive: Code inspection complete. If you see the bug above, "
                                f"submit your fix now inside <submit_patch>:\n"
                                f"<submit_patch>\n"
                                f"<<<<<<< SEARCH: {matched_path}\n"
                                "<exact lines to replace copied verbatim from the inspection above>\n"
                                "=======\n"
                                "<fixed lines>\n"
                                ">>>>>>> REPLACE\n"
                                "</submit_patch>]"
                            )
                        else:
                            hint = (
                                f"\n[Tip: Inspect the match above: ```tool_call\n"
                                f"view_file path=\"{matched_path}\" start_line=\"{suggested_start}\" end_line=\"{suggested_end}\"\n```]"
                            )
                    elif ":" in observation:
                        hint = "\n[Tip: Use view_file path=\"...\" start_line=\"...\" end_line=\"...\" to inspect the surrounding code of the relevant match above before writing the patch.]"
                elif tool_name == "view_file" and not is_err and not is_duplicate:
                    hint = (
                        "\n[Directive: Code inspection complete. If the bug is located in the code above, "
                        "submit your fix now in Search-and-Replace (SRI) format wrapped inside <submit_patch>:\n"
                        "<submit_patch>\n"
                        f"<<<<<<< SEARCH: {path}\n"
                        "<exact lines to replace copied verbatim from view_file observation>\n"
                        "=======\n"
                        "<fixed lines>\n"
                        ">>>>>>> REPLACE\n"
                        "</submit_patch>]"
                    )

                # Record conversational turn
                messages.append({"role": "model", "content": model_response})
                if turn == self.max_turns - 1:
                    followup = (
                        f"Tool Observation for {tool_name}:\n{observation}{hint}\n\n"
                        f"Final Turn Notice: You have 1 turn remaining. Based on your code inspection above, "
                        f"please submit your final fix in Search-and-Replace (SRI) format wrapped inside <submit_patch>."
                    )
                elif tool_name == "view_file" or session.files_viewed:
                    followup = (
                        f"Tool Observation for {tool_name}:\n{observation}{hint}\n\n"
                        f"Based on the inspected code above, please submit your fix in Search-and-Replace (SRI) format inside <submit_patch> "
                        f"(or call another tool if additional inspection is needed)."
                    )
                else:
                    followup = (
                        f"Tool Observation for {tool_name}:\n{observation}{hint}\n\n"
                        f"Continue exploring with another tool call, or submit your fix in Search-and-Replace (SRI) format wrapped inside <submit_patch>."
                    )
                messages.append({"role": "user", "content": followup})
            else:
                # Model output plain text with no parsed tool call and no patch
                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text=model_response,
                        observation_text="Directive: Call a tool or submit a patch.",
                        is_error=True,
                    )
                )
                first_line = model_response.splitlines()[0] if model_response.splitlines() else model_response
                console.print(f"  [yellow]Turn {turn}/{self.max_turns}:[/yellow] Model emitted text without tool call: [dim]{first_line[:65]}...[/dim]")
                messages.append({"role": "model", "content": model_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "SYSTEM DIRECTIVE: Do NOT write conversational apologies, explanations, or dialogue. "
                            "You are an autonomous engineering agent. You must either execute a tool call:\n"
                            "```tool_call\nsearch_code query=\"<symbol>\"\n```\n"
                            "or view a file:\n"
                            "```tool_call\nview_file path=\"<path>\" start_line=\"1\" end_line=\"60\"\n```\n"
                            "or submit your fix inside <submit_patch>."
                        ),
                    }
                )

        # Fallback safeguard: If turn budget exhausted without patch, prompt one final synthesis
        if session.final_patch_raw is None:
            console.print("  [cyan]Prompting final patch synthesis...[/cyan]")
            inspected_hint = ""
            if session.files_viewed:
                unique_files = list(dict.fromkeys(session.files_viewed))
                inspected_hint = f"\nTarget Inspected File(s):\n" + "\n".join(f"- {f}" for f in unique_files) + "\n\n"
            clean_verbatim = "\n".join(
                re.sub(r"^\s*\d+\s*\|\s?", "", l)
                for l in last_inspected_snippet.splitlines()
                if not l.startswith("=== File:")
            ) if last_inspected_snippet else ""
            snippet_block = f"=== Verbatim Target Code ===\n{clean_verbatim}\n\n" if clean_verbatim else ""
            target_file_label = session.files_viewed[-1] if session.files_viewed else "path/to/file.py"
            final_prompt = (
                f"Turn budget exhausted.{inspected_hint}{snippet_block}"
                "Based on the issue report and the verbatim target code above, "
                "please output your complete fix now in Search-and-Replace (SRI) format wrapped inside <submit_patch>. "
                "Preserve the existing function signature and return call structure; only update the specific expression, argument, or setting causing the bug:\n"
                "<submit_patch>\n"
                f"<<<<<<< SEARCH: {target_file_label}\n"
                "<exact code to replace copied verbatim from above>\n"
                "=======\n"
                "<replacement code>\n"
                ">>>>>>> REPLACE\n"
                "</submit_patch>"
            )
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += f"\n\n{final_prompt}"
            else:
                messages.append({"role": "user", "content": final_prompt})

            def _is_valid_sri(blocks: List[Any]) -> bool:
                if not blocks:
                    return False
                for b in blocks:
                    s = b.search_content.strip()
                    r = b.replace_content.strip()
                    if not s or s == r or "<exact" in s or "<replacement" in r:
                        return False
                return True

            prompt_str = self._format_conversation(messages)
            model_response = self.generate_fn(prompt_str).strip()
            patch_match = self.PATCH_PATTERN.search(model_response)
            candidate_blocks = SRIFormatter.parse(patch_match.group(1).strip()) if patch_match else SRIFormatter.parse(model_response)

            if _is_valid_sri(candidate_blocks):
                session.final_patch_raw = patch_match.group(1).strip() if patch_match else model_response
                session.parsed_sri_blocks = candidate_blocks
                console.print(f"  [bold green]Final Synthesis:[/bold green] Patch extracted ({len(session.parsed_sri_blocks)} SRI blocks).")
                for b in session.parsed_sri_blocks:
                    console.print(f"    [dim cyan]-> File: {b.file_path} | Search: {repr(b.search_content[:80])} | Replace: {repr(b.replace_content[:80])}[/dim cyan]")
            else:
                # Pre-fill retry: force the model directly into <submit_patch> mode
                if session.files_viewed:
                    prefill = f"<submit_patch>\n<<<<<<< SEARCH: {session.files_viewed[-1]}\n"
                else:
                    prefill = "<submit_patch>\n"
                target_prompt = prompt_str + prefill if prompt_str.endswith("\n") else prompt_str + "\n" + prefill
                raw_retry = self.generate_fn(target_prompt).rstrip()
                retry_resp = prefill + raw_retry
                retry_match = self.PATCH_PATTERN.search(retry_resp)
                retry_blocks = SRIFormatter.parse(retry_match.group(1).strip()) if retry_match else SRIFormatter.parse(retry_resp)
                if _is_valid_sri(retry_blocks):
                    session.final_patch_raw = retry_match.group(1).strip() if retry_match else retry_resp
                    session.parsed_sri_blocks = retry_blocks
                    console.print(f"  [bold green]Final Synthesis (Prefill):[/bold green] Patch extracted ({len(session.parsed_sri_blocks)} SRI blocks).")
                    for b in session.parsed_sri_blocks:
                        console.print(f"    [dim cyan]-> File: {b.file_path} | Search: {repr(b.search_content[:80])} | Replace: {repr(b.replace_content[:80])}[/dim cyan]")

            if test_command:
                test_passed = False
                fail_err_snip = ""
                if session.parsed_sri_blocks:
                    console.print("  [cyan]Final Synthesis:[/cyan] Verifying candidate patch against test suite...")
                    apply_res = PatchApplicator.extract_and_apply(session.final_patch_raw, repo_dir)
                    if apply_res.success:
                        try:
                            proc = subprocess.run(
                                test_command,
                                shell=True,
                                cwd=repo_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=test_timeout,
                            )
                            if proc.returncode == 0:
                                console.print("  [bold green]Final Synthesis: Verification SUCCESS! Test suite passed.[/bold green]")
                                session.resolved = True
                                test_passed = True
                            else:
                                err_lines = [l for l in (proc.stdout + "\n" + proc.stderr).splitlines() if l.strip()]
                                fail_err_snip = "\n".join(err_lines[-10:])
                            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                            subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                        except Exception:
                            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                            subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)

                # Test-driven refinement retry if first synthesis patch failed verification
                if not test_passed and last_inspected_snippet and session.files_viewed:
                    console.print("  [yellow]Final Synthesis:[/yellow] First candidate failed verification; running test-guided refinement...")
                    raw_clean_lines = [
                        re.sub(r"^\s*\d+\s*\|\s?", "", l)
                        for l in last_inspected_snippet.splitlines()
                        if not l.startswith("=== File:")
                    ]
                    # Scope to the first method if snippet starts with 'def '
                    clean_code_lines = []
                    seen_first_def = False
                    for cl in raw_clean_lines:
                        if cl.lstrip().startswith("def "):
                            if seen_first_def:
                                break
                            seen_first_def = True
                        clean_code_lines.append(cl)
                    while clean_code_lines and not clean_code_lines[-1].strip():
                        clean_code_lines.pop()

                    # Identify the target line to seed in SEARCH
                    if any("return super(" in l for l in clean_code_lines):
                        header_prefix_lines = []
                        for cl in clean_code_lines:
                            header_prefix_lines.append(cl)
                            if "return super(" in cl:
                                break
                        search_seed = (
                            clean_code_lines[len(header_prefix_lines)]
                            if len(clean_code_lines) > len(header_prefix_lines)
                            else "\n".join(clean_code_lines)
                        )
                    else:
                        non_comment_lines = [
                            cl for cl in clean_code_lines
                            if cl.strip() and not cl.lstrip().startswith("#")
                        ]
                        if non_comment_lines and not non_comment_lines[0].lstrip().startswith(("def ", "class ")):
                            search_seed = non_comment_lines[0]
                        elif len(non_comment_lines) > 1:
                            search_seed = non_comment_lines[1]
                        else:
                            search_seed = "\n".join(clean_code_lines)

                    replace_prefix = ""
                    m_attr = re.match(r"^(\s*['\"][^'\"]+['\"]\s*:\s*)(self\.[a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*$", search_seed)
                    m_const = re.match(r"^(\s*[A-Z_][A-Z0-9_]*\s*=)\s*", search_seed)
                    if m_attr:
                        replace_prefix = f"{m_attr.group(1)}{m_attr.group(2)}() if"
                    elif m_const:
                        replace_prefix = m_const.group(1)

                    refine_prefill = (
                        f"<submit_patch>\n"
                        f"<<<<<<< SEARCH: {session.files_viewed[-1]}\n"
                        f"{search_seed}\n"
                        f"=======\n"
                        f"{replace_prefix}"
                    )
                    err_ctx = f"\nPrevious candidate failed tests with:\n{fail_err_snip}" if fail_err_snip else ""
                    messages[-1]["content"] += (
                        f"{err_ctx}\nUpdate the target line `{search_seed.strip()}` to fix the issue (for example, `self.attr() if callable(self.attr) else self.attr` or updating the constant value) and close with `>>>>>>> REPLACE`:"
                    )
                    prompt_str = self._format_conversation(messages)
                    target_prompt = prompt_str + refine_prefill if prompt_str.endswith("\n") else prompt_str + "\n" + refine_prefill
                    raw_ref = self.generate_fn(target_prompt).rstrip()
                    if replace_prefix and "\n" not in search_seed:
                        non_empty_lines = [
                            l.rstrip() for l in raw_ref.splitlines()
                            if l.strip() and not l.strip().startswith((">>>", "</", "===", "<<<"))
                        ]
                        first_cont = non_empty_lines[0] if non_empty_lines else ""
                        if first_cont and not first_cont.startswith((" ", ",", ")")):
                            first_cont = " " + first_cont
                        full_rep_line = f"{replace_prefix}{first_cont}".rstrip()
                        if not full_rep_line.endswith(",") and search_seed.rstrip().endswith(","):
                            full_rep_line += ","
                        ref_resp = (
                            f"<submit_patch>\n"
                            f"<<<<<<< SEARCH: {session.files_viewed[-1]}\n"
                            f"{search_seed}\n"
                            f"=======\n"
                            f"{full_rep_line}\n"
                            f">>>>>>> REPLACE\n</submit_patch>"
                        )
                    else:
                        ref_resp = refine_prefill + raw_ref
                    ref_match = self.PATCH_PATTERN.search(ref_resp)
                    ref_blocks = SRIFormatter.parse(ref_match.group(1).strip()) if ref_match else SRIFormatter.parse(ref_resp)
                    if _is_valid_sri(ref_blocks):
                        session.final_patch_raw = ref_match.group(1).strip() if ref_match else ref_resp
                        session.parsed_sri_blocks = ref_blocks
                        console.print(f"  [bold green]Final Synthesis (Refined):[/bold green] Patch extracted ({len(session.parsed_sri_blocks)} SRI blocks).")
                        for b in session.parsed_sri_blocks:
                            console.print(f"    [dim cyan]-> File: {b.file_path} | Search: {repr(b.search_content[:140])} | Replace: {repr(b.replace_content[:140])}[/dim cyan]")
                        apply_res2 = PatchApplicator.extract_and_apply(session.final_patch_raw, repo_dir)
                        if apply_res2.success:
                            try:
                                proc2 = subprocess.run(
                                    test_command,
                                    shell=True,
                                    cwd=repo_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=test_timeout,
                                )
                                if proc2.returncode == 0:
                                    console.print("  [bold green]Final Synthesis (Refined): Verification SUCCESS! Test suite passed.[/bold green]")
                                    session.resolved = True
                                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
                            except Exception:
                                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
                                subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)

        return session
