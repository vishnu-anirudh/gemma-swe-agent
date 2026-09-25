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


def test_sri_apply_with_line_number_prefixes():
    content = "def calculate(x):\n    result = x * 2\n    return result\n"
    search = "  42 |     result = x * 2\n  43 |     return result"
    replace = "  42 |     result = x * 3\n  43 |     return result"

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is True
    assert "x * 3" in new_content
    assert "x * 2" not in new_content


def test_sri_apply_with_fuzzy_matching():
    content = "def separability_matrix(transform):\n    if hasattr(transform, '_calculate_separability_matrix'):\n        return transform._calculate_separability_matrix()\n    return _separable(transform)\n"
    search = "def separability_matrix(transform):\n    if hasattr(transform, '_calculate_separability_matrix'):"
    replace = "def separability_matrix(transform):\n    # Fix applied\n    if hasattr(transform, '_calculate_separability_matrix'):"

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is True
    assert "# Fix applied" in new_content


def test_sri_apply_with_sub_chunk_matching():
    content = (
        "def separability_matrix(transform):\n"
        '    """Docstring explaining function."""\n'
        "    separable_matrix = _separable(transform)\n"
        "    is_separable = separable_matrix.sum(1)\n"
        "    return is_separable\n"
    )
    # Search block has anchor header that is separated by docstring in target file
    search = (
        "def separability_matrix(transform):\n"
        "    separable_matrix = _separable(transform)\n"
        "    is_separable = separable_matrix.sum(1)"
    )
    replace = (
        "def separability_matrix(transform):\n"
        "    # Custom hook\n"
        "    separable_matrix = _separable(transform)\n"
        "    is_separable = separable_matrix.sum(1)"
    )

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is True
    assert "# Custom hook" in new_content


def test_sri_apply_with_anchor_scope_matching():
    content = (
        "import numpy as np\n\n"
        "def _separable(transform):\n"
        '    """Docstring explaining function."""\n'
        "    if transform_matrix is not NotImplemented:\n"
        "        return transform_matrix\n"
        "    elif isinstance(transform, CompoundModel):\n"
        "        sepleft = _separable(transform.left)\n"
        "        return sepleft\n"
    )
    # Search block has anchor header, comment, and 'if' instead of 'elif'
    search = (
        "def _separable(transform):\n"
        "    if isinstance(transform, CompoundModel):\n"
        "        # Handle nested CompoundModels here"
    )
    replace = (
        "def _separable(transform):\n"
        "    if isinstance(transform, CompoundModel):\n"
        "        # Handle nested CompoundModels here\n"
        "        sepleft = _separable(transform.left)\n"
        "        sepright = _separable(transform.right)\n"
        "        return _operators[transform.op](sepleft, sepright)"
    )

    new_content, success, msg = SRIFormatter.apply_to_content(content, search, replace)
    assert success is True
    assert "sepright = _separable(transform.right)" in new_content


def test_sri_nested_marker_cleaning():
    raw_text = """
<<<<<<< SEARCH: astropy/io/ascii/rst.py
<<<<<<< SEARCH: def get_fixedwidth_params(self, line):
=======
<<<<<<< REPLACE: def get_fixedwidth_params(self, line, header_rows=None):
>>>>>>> REPLACE
"""
    blocks = SRIFormatter.parse(raw_text)
    assert len(blocks) == 1
    assert blocks[0].file_path == "astropy/io/ascii/rst.py"
    assert blocks[0].search_content == "def get_fixedwidth_params(self, line):"
    assert "header_rows=None" in blocks[0].replace_content

    content = "class RST:\n    def get_fixedwidth_params(self, line):\n        pass\n"
    new_content, success, msg = SRIFormatter.apply_to_content(content, blocks[0].search_content, blocks[0].replace_content)
    assert success is True
    assert "header_rows=None" in new_content




