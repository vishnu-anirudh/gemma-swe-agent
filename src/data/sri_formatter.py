"""Search-and-Replace Infilling (SRI) Formatter & Patch Engine.

Implements the SRI paradigm described in research.md for robust,
line-number-free code edits across repository files.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SRIBlock:
    """Represents a single search-and-replace edit block."""

    file_path: str
    search_content: str
    replace_content: str

    def is_valid(self) -> bool:
        """Check if the search block has a valid file path."""
        return bool(self.file_path.strip())


class SRIParseError(ValueError):
    """Raised when an SRI block cannot be parsed properly."""

    pass


class SRIApplicationError(RuntimeError):
    """Raised when an SRI block fails to match or apply cleanly."""

    pass


class SRIFormatter:
    """Parses, validates, applies, and formats Search-and-Replace Infilling (SRI) blocks."""

    # Matches patterns like:
    # <<<<<<< SEARCH: path/to/file.py
    # ...
    # =======
    # ...
    # >>>>>>> REPLACE
    #
    # Also handles variations with markdown backticks or without space after colon
    BLOCK_PATTERN = re.compile(
        r"<<<<<<<\s*SEARCH(?::\s*([^\n\r]+))?\s*\r?\n"
        r"(.*?)\r?\n"
        r"=======\s*\r?\n"
        r"(.*?)\r?\n"
        r">>>>>>>\s*REPLACE",
        re.DOTALL,
    )

    @classmethod
    def parse(cls, text: str, default_file_path: Optional[str] = None) -> List[SRIBlock]:
        """Parse all SRI blocks from model output text.

        Args:
            text: Raw string containing model generation.
            default_file_path: Fallback path if block omits explicit file name.

        Returns:
            List of parsed SRIBlock instances.
        """
        blocks: List[SRIBlock] = []

        # Remove surrounding markdown wrappers if full text is wrapped in a code fence
        clean_text = text.strip()
        if clean_text.startswith("```") and clean_text.endswith("```"):
            lines = clean_text.splitlines()
            if len(lines) >= 2:
                clean_text = "\n".join(lines[1:-1])

        matches = cls.BLOCK_PATTERN.finditer(clean_text)
        current_file = default_file_path

        for match in matches:
            file_header, search_content, replace_content = match.groups()
            file_path = file_header.strip() if file_header else current_file

            if not file_path:
                raise SRIParseError(
                    "SRI Block missing file target and no default_file_path was provided."
                )

            current_file = file_path
            blocks.append(
                SRIBlock(
                    file_path=file_path,
                    search_content=search_content,
                    replace_content=replace_content,
                )
            )

        return blocks

    @classmethod
    def apply_to_content(
        cls, original_content: str, search_str: str, replace_str: str
    ) -> Tuple[str, bool, str]:
        """Apply a single search-and-replace operation on a string.

        Requires exact unique matching to prevent ambiguous modifications.

        Returns:
            Tuple of (new_content, success, message)
        """
        # Exact match count
        count = original_content.count(search_str)

        if count == 1:
            new_content = original_content.replace(search_str, replace_str, 1)
            return new_content, True, "Applied successfully."

        if count == 0:
            # Fallback 1: Normalize newline line endings
            norm_content = original_content.replace("\r\n", "\n")
            norm_search = search_str.replace("\r\n", "\n")
            norm_replace = replace_str.replace("\r\n", "\n")
            norm_count = norm_content.count(norm_search)

            if norm_count == 1:
                new_content = norm_content.replace(norm_search, norm_replace, 1)
                return new_content, True, "Applied with normalized newlines."

            # Fallback 2: Strip trailing whitespaces per line
            lines_content = [line.rstrip() for line in norm_content.splitlines()]
            lines_search = [line.rstrip() for line in norm_search.splitlines()]

            joined_content = "\n".join(lines_content)
            joined_search = "\n".join(lines_search)
            if joined_content.count(joined_search) == 1:
                # Replace maintaining general line count
                replaced_joined = joined_content.replace(joined_search, norm_replace, 1)
                return (
                    replaced_joined,
                    True,
                    "Applied with whitespace-relaxed matching.",
                )

            return original_content, False, "Search string not found in target file."

        return (
            original_content,
            False,
            f"Search string matched {count} locations (must match exactly once).",
        )

    @classmethod
    def apply_blocks_to_files(
        cls, file_contents: Dict[str, str], blocks: List[SRIBlock]
    ) -> Tuple[Dict[str, str], List[str]]:
        """Apply an ordered list of SRI blocks to multiple file contents.

        Args:
            file_contents: Dict mapping relative file_path -> current string content.
            blocks: List of SRIBlock to apply.

        Returns:
            Tuple of (updated_file_contents_dict, errors_list)
        """
        updated_files = dict(file_contents)
        errors: List[str] = []

        for idx, block in enumerate(blocks):
            if block.file_path not in updated_files:
                errors.append(
                    f"Block {idx+1}: File '{block.file_path}' not present in provided files."
                )
                continue

            current_content = updated_files[block.file_path]
            new_content, success, msg = cls.apply_to_content(
                current_content, block.search_content, block.replace_content
            )

            if not success:
                errors.append(
                    f"Block {idx+1} for '{block.file_path}' failed: {msg}"
                )
            else:
                updated_files[block.file_path] = new_content

        return updated_files, errors

    @classmethod
    def to_unified_diff(
        cls, original_files: Dict[str, str], modified_files: Dict[str, str]
    ) -> str:
        """Convert changes between original and modified files into a unified git diff.

        Useful for piping to git apply or the official SWE-bench evaluation harness.
        """
        diff_chunks: List[str] = []

        for file_path, orig in original_files.items():
            mod = modified_files.get(file_path, orig)
            if orig == mod:
                continue

            orig_lines = orig.splitlines(keepends=True)
            mod_lines = mod.splitlines(keepends=True)

            diff = difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
            diff_chunks.append("".join(diff))

        return "".join(diff_chunks)

    @classmethod
    def format_block(cls, file_path: str, search_content: str, replace_content: str) -> str:
        """Helper to generate a formatted SRI string block."""
        return (
            f"<<<<<<< SEARCH: {file_path}\n"
            f"{search_content}\n"
            f"=======\n"
            f"{replace_content}\n"
            f">>>>>>> REPLACE"
        )
