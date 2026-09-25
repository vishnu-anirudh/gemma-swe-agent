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
        "You are an expert autonomous software engineering agent. You are tasked with resolving a bug in a codebase.\n"
        "You can inspect files, search for symbols, and run tests before submitting your fix.\n\n"
        "Available Tool Calls:\n"
        "- <tool_call name=\"view_file\" path=\"<rel_path>\" start_line=\"1\" end_line=\"50\" />\n"
        "- <tool_call name=\"search_code\" query=\"<string>\" />\n"
        "- <tool_call name=\"list_dir\" path=\"<dir_path>\" />\n"
        "- <tool_call name=\"run_test\" command=\"<test_cmd>\" />\n\n"
        "When you have identified the fix, submit it in Search-and-Replace Infilling (SRI) format wrapped inside:\n"
        "<submit_patch>\n"
        "<<<<<<< SEARCH: relative/path/to/file.py\n"
        "<exact original code to search for>\n"
        "=======\n"
        "<replacement code>\n"
        ">>>>>>> REPLACE\n"
        "</submit_patch>\n\n"
        "Always inspect the relevant code first. Search blocks MUST match the actual file contents exactly."
    )

    TOOL_PATTERN = re.compile(
        r'<tool_call\s+name="([^"]+)"(?:\s+path="([^"]*)")?(?:\s+query="([^"]*)")?(?:\s+command="([^"]*)")?(?:\s+start_line="(\d+)")?(?:\s+end_line="(\d+)")?\s*/>'
    )
    PATCH_PATTERN = re.compile(r"<submit_patch>(.*?)</submit_patch>", re.DOTALL)

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        max_turns: int = 6,
    ):
        self.generate_fn = generate_fn
        self.max_turns = max_turns

    def run_session(
        self,
        instance_id: str,
        problem_statement: str,
        repo_dir: str,
    ) -> AgentSession:
        """Execute a multi-turn exploration and bug-fixing session."""
        session = AgentSession(instance_id=instance_id, turns=0, tool_calls_made=0)

        history = (
            f"{self.AGENT_SYSTEM_PROMPT}\n\n"
            f"=== Issue Report for {instance_id} ===\n"
            f"{problem_statement}\n\n"
            f"Begin by listing directories or searching for files related to this issue."
        )

        for turn in range(1, self.max_turns + 1):
            session.turns = turn
            model_response = self.generate_fn(history).strip()

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

            # 2. Check for tool calls
            tool_match = self.TOOL_PATTERN.search(model_response)
            if tool_match:
                tool_name = tool_match.group(1)
                path = tool_match.group(2) or "."
                query = tool_match.group(3) or ""
                command = tool_match.group(4) or ""
                start_l = int(tool_match.group(5) or 1)
                end_l = int(tool_match.group(6) or 100)

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

                # Append to context for next turn
                history += (
                    f"\n\n<assistant_turn_{turn}>\n{model_response}\n</assistant_turn_{turn}>\n"
                    f"<observation>\n{observation}\n</observation>\n"
                    f"Continue exploring or submit your fix using <submit_patch>."
                )
            else:
                # Model output plain text with no tool call and no patch
                session.trajectory_steps.append(
                    TrajectoryStep(
                        action_text=model_response,
                        observation_text="Notice: Please call a tool or submit a patch.",
                        is_error=True,
                    )
                )
                history += (
                    f"\n\n<assistant_turn_{turn}>\n{model_response}\n</assistant_turn_{turn}>\n"
                    f"<observation>Notice: No tool call detected. Please call a tool or submit a patch.</observation>"
                )

        return session
