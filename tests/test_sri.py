"""Unit tests for Search-and-Replace Infilling (SRI) Formatter."""

import pytest
from src.data.sri_formatter import SRIBlock, SRIFormatter, SRIParseError


def test_sri_parsing():
    raw_text = """
Here is the fix for the bug:
<<<<<<< SEARCH: src/calculator.py
def add(a, b):
    return a - b
=======
def add(a, b):
    return a + b
>>>>>>> REPLACE
End of solution.
"""
    blocks = SRIFormatter.parse(raw_text)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.file_path == "src/calculator.py"
    assert "return a - b" in block.search_content
    assert "return a + b" in block.replace_content


def test_sri_apply_to_content():
    content = "def hello():\n    print('goodbye')\n"
    search = "    print('goodbye')"
    replace = "    print('hello world')"

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is True
    assert "print('hello world')" in new_content
    assert "print('goodbye')" not in new_content


def test_sri_non_unique_match_fails():
    content = "val = 1\nval = 1\n"
    search = "val = 1"
    replace = "val = 2"

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is False
    assert "matched 2 locations" in msg
    assert new_content == content


def test_sri_to_unified_diff():
    orig = {"file.py": "def foo():\n    return 1\n"}
    mod = {"file.py": "def foo():\n    return 2\n"}

    diff = SRIFormatter.to_unified_diff(orig, mod)
    assert "--- a/file.py" in diff
    assert "+++ b/file.py" in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff
