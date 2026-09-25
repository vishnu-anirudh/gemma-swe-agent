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
            # Strip accidental nested markers (e.g. '<<<<<<< SEARCH: ...')
            cleaned_search = re.sub(
                r"^<{3,7}\s*(?:SEARCH|REPLACE):?\s*", "", search_content.strip(), flags=re.IGNORECASE
            )
            cleaned_replace = re.sub(
                r"^<{3,7}\s*(?:SEARCH|REPLACE):?\s*", "", replace_content.strip(), flags=re.IGNORECASE
            )
            blocks.append(
                SRIBlock(
                    file_path=file_path,
                    search_content=cleaned_search,
                    replace_content=cleaned_replace,
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
        # Clean residual nested markers
        search_str = re.sub(
            r"^<{3,7}\s*(?:SEARCH|REPLACE):?\s*", "", search_str.strip(), flags=re.IGNORECASE
        )
        replace_str = re.sub(
            r"^<{3,7}\s*(?:SEARCH|REPLACE):?\s*", "", replace_str.strip(), flags=re.IGNORECASE
        )

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

            # Fallback 3: Strip tool observation line-number prefixes (e.g. '  66 | def ...')
            cleaned_search = "\n".join(re.sub(r"^\s*\d+\s*\|\s?", "", l) for l in norm_search.splitlines())
            cleaned_replace = "\n".join(re.sub(r"^\s*\d+\s*\|\s?", "", l) for l in norm_replace.splitlines())
            if norm_content.count(cleaned_search) == 1:
                new_content = norm_content.replace(cleaned_search, cleaned_replace, 1)
                return new_content, True, "Applied with line-number prefix stripping."

            # Fallback 4: Line-number prefix stripping + trailing whitespace relaxation
            lines_cleaned_search = [re.sub(r"^\s*\d+\s*\|\s?", "", l).rstrip() for l in lines_search]
            joined_cleaned_search = "\n".join(lines_cleaned_search)
            if joined_content.count(joined_cleaned_search) == 1:
                replaced_joined = joined_content.replace(joined_cleaned_search, cleaned_replace, 1)
                return (
                    replaced_joined,
                    True,
                    "Applied with line-number prefix stripping and whitespace relaxation.",
                )

            # Fallback 5: Fuzzy window matching for block replacement
            search_lines = [l.strip() for l in cleaned_search.splitlines() if l.strip()]
            if len(search_lines) >= 2:
                content_lines = norm_content.splitlines()
                match_indices = []
                first_target = search_lines[0]
                for i, c_line in enumerate(content_lines):
                    if first_target in c_line.strip() or c_line.strip().startswith(first_target[:20]):
                        matched = 0
                        for j, s_line in enumerate(search_lines):
                            if i + j < len(content_lines) and s_line in content_lines[i + j].strip():
                                matched += 1
                        if matched >= max(2, int(len(search_lines) * 0.7)):
                            match_indices.append((i, i + len(search_lines)))

                if len(match_indices) == 1:
                    start_i, end_i = match_indices[0]
                    new_lines = content_lines[:start_i] + cleaned_replace.splitlines() + content_lines[end_i:]
                    return "\n".join(new_lines), True, "Applied with fuzzy window matching."

            # Fallback 6: Contiguous sub-chunk alignment matching
            # Handles cases where model included an anchor line separated by docstrings
            s_raw_lines = cleaned_search.splitlines()
            r_raw_lines = cleaned_replace.splitlines()
            best_chunk_match = None
            if len(s_raw_lines) >= 2:
                for s_start in range(len(s_raw_lines)):
                    for s_end in range(len(s_raw_lines), s_start + 1, -1):
                        sub_chunk = "\n".join(s_raw_lines[s_start:s_end])
                        if len(sub_chunk.strip()) > 15 and norm_content.count(sub_chunk) == 1:
                            best_chunk_match = (s_start, s_end, sub_chunk)
                            break
                    if best_chunk_match:
                        break

            if best_chunk_match:
                s_start, s_end, sub_chunk = best_chunk_match
                if r_raw_lines[:s_start] == s_raw_lines[:s_start]:
                    sub_replace = "\n".join(r_raw_lines[s_start:])
                else:
                    sub_replace = cleaned_replace
                new_content = norm_content.replace(sub_chunk, sub_replace, 1)
                return new_content, True, "Applied with contiguous sub-chunk alignment matching."

            # Fallback 7: Anchor function scope matching with keyword/comment relaxation
            # If search begins with a function/class header, locate that entity's scope,
            # and search for the core non-comment statement inside that scope.
            non_comment_s = [
                l for l in s_raw_lines if l.strip() and not l.strip().startswith("#")
            ]
            if len(non_comment_s) >= 2:
                header_line = non_comment_s[0].strip()
                header_match = re.search(r"^(?:def|class)\s+([a-zA-Z0-9_]+)", header_line)
                if header_match:
                    entity_name = header_match.group(1)
                    entity_pattern = re.compile(
                        rf"\b(?:def|class)\s+{re.escape(entity_name)}\s*[\(:]"
                    )
                    entity_found = entity_pattern.search(norm_content)
                    if entity_found:
                        start_pos = entity_found.start()
                        scope_text = norm_content[start_pos : start_pos + 3000]

                        target_statement = non_comment_s[1].strip()
                        relaxed_statement = re.sub(
                            r"^(?:elif|if)\s+", "", target_statement
                        )

                        if len(relaxed_statement) > 10 and scope_text.count(relaxed_statement) == 1:
                            exact_target_line = None
                            for line in scope_text.splitlines():
                                if relaxed_statement in line:
                                    exact_target_line = line
                                    break

                            if exact_target_line:
                                r_body_lines = [
                                    l
                                    for l in r_raw_lines
                                    if not l.strip().startswith(header_match.group(0))
                                ]
                                clean_sub_replace = "\n".join(r_body_lines)
                                new_content = norm_content.replace(
                                    exact_target_line, clean_sub_replace, 1
                                )
                                return (
                                    new_content,
                                    True,
                                    f"Applied via anchor scope matching in {entity_name}.",
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
