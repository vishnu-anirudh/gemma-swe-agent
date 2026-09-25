"""Interactive Multi-Turn Agent Loop for Repository-Level Bug Fixing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.tools import RepoTools
from src.data.masking import TrajectoryStep
from src.data.sri_formatter import SRIBlock, SRIFormatter


@dataclass
class AgentSession:
    """Records the trajectory, interaction history, and final patch of an agent run."""

    instance_id: str
    turns: int
    tool_calls_made: int
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    final_patch_raw: Optional[str] = None
    parsed_sri_blocks: List[SRIBlock] = field(default_factory=list)
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
        "<exact original code to search for>\n"
        "=======\n"
        "<replacement code>\n"
        ">>>>>>> REPLACE\n"
        "</submit_patch>\n\n"
        "Always inspect the relevant code first. Search blocks MUST match the actual file contents exactly."
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

        # 2. Markdown block: ```tool_call ... ```
        block_match = re.search(r'```(?:tool_call|tool)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if block_match:
            inner = block_match.group(1).strip()
            # Check for inner XML
            xml_inner = re.search(r'<tool_call\s+name="([^"]+)"([^>]*)/?>', inner)
            if xml_inner:
                tool_name = xml_inner.group(1).strip()
                attrs = dict(re.findall(r'(\w+)="([^"]*)"', xml_inner.group(2)))
                return tool_name, attrs

            parts = inner.split(maxsplit=1)
            if parts:
                tool_name = parts[0].strip().replace("()", "")
                rest = parts[1] if len(parts) > 1 else ""
                clean_attrs = {}
                for m in re.finditer(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s\)]+))', rest):
                    key = m.group(1)
                    val = m.group(2) or m.group(3) or m.group(4) or ""
                    clean_attrs[key] = val
                if tool_name == "search_code" and "name" in clean_attrs and "query" not in clean_attrs:
                    clean_attrs["query"] = clean_attrs["name"]
                return tool_name, clean_attrs

        # 3. Direct line: search_code query="..." or view_file path="..."
        direct_match = re.search(r'\b(search_code|view_file|list_dir|run_test)\b\s*\(?([^)\n]*)\)?', text)
        if direct_match:
            tool_name = direct_match.group(1)
            rest = direct_match.group(2)
            clean_attrs = {}
            for m in re.finditer(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|([^\s\)]+))', rest):
                key = m.group(1)
                val = m.group(2) or m.group(3) or m.group(4) or ""
                clean_attrs[key] = val
            if tool_name == "search_code" and "name" in clean_attrs and "query" not in clean_attrs:
                clean_attrs["query"] = clean_attrs["name"]
            return tool_name, clean_attrs

        return None

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using tokenizer chat template if available, else plain text."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback to standard dialogue format
        rendered = []
        for msg in messages:
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
    ) -> AgentSession:
        """Execute a multi-turn exploration and bug-fixing session."""
        session = AgentSession(instance_id=instance_id, turns=0, tool_calls_made=0)

        initial_user_msg = (
            f"{self.AGENT_SYSTEM_PROMPT}\n\n"
            f"=== Issue Report for {instance_id} ===\n"
            f"{problem_statement}\n\n"
            f"Please begin by searching the codebase or viewing relevant files using a tool call."
        )

        messages = [{"role": "user", "content": initial_user_msg}]

        for turn in range(1, self.max_turns + 1):
            session.turns = turn
            prompt_str = self._format_conversation(messages)
            model_response = self.generate_fn(prompt_str).strip()

            # 1. Check if the model submitted a final patch
            patch_match = self.PATCH_PATTERN.search(model_response)
            if patch_match:
                patch_text = patch_match.group(1).strip()
                session.final_patch_raw = patch_text
                session.parsed_sri_blocks = SRIFormatter.parse(patch_text)
                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text="submit_patch",
                        observation_text="Patch submitted.",
                        is_error=False,
                    )
                )
                break

            # Or check if raw SRI blocks were emitted directly
            sri_blocks = SRIFormatter.parse(model_response)
            if len(sri_blocks) > 0:
                session.final_patch_raw = model_response
                session.parsed_sri_blocks = sri_blocks
                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text="submit_patch_raw",
                        observation_text="Direct SRI patch detected.",
                        is_error=False,
                    )
                )
                break

            # 2. Check for tool calls using multi-format parser
            parsed = self.parse_tool_call(model_response)
            if parsed:
                tool_name, attrs = parsed
                path = attrs.get("path", ".")
                query = attrs.get("query", attrs.get("name", ""))
                command = attrs.get("command", "")
                start_l = int(attrs.get("start_line", 1))
                end_l = int(attrs.get("end_line", 100))

                session.tool_calls_made += 1

                if tool_name == "view_file":
                    observation = RepoTools.view_file(repo_dir, path, start_l, end_l)
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

                # Record conversational turn
                messages.append({"role": "model", "content": model_response})
                if turn == self.max_turns - 1:
                    followup = (
                        f"Tool Observation for {tool_name}:\n{observation}\n\n"
                        f"Final Turn Notice: You have 1 turn remaining. Based on your code inspection above, "
                        f"please submit your final fix in Search-and-Replace (SRI) format wrapped inside <submit_patch>."
                    )
                else:
                    followup = (
                        f"Tool Observation for {tool_name}:\n{observation}\n\n"
                        f"Continue exploring with another tool call, or submit your fix in Search-and-Replace (SRI) format wrapped inside <submit_patch>."
                    )
                messages.append({"role": "user", "content": followup})
            else:
                # Model output plain text with no parsed tool call and no patch
                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text=model_response,
                        observation_text="Notice: Please call a tool or submit a patch.",
                        is_error=True,
                    )
                )
                messages.append({"role": "model", "content": model_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Notice: No tool call detected. Please call a tool (e.g. ```tool_call\nsearch_code query=\"...\"\n```) "
                            "or submit your fix wrapped inside <submit_patch>."
                        ),
                    }
                )

        return session
