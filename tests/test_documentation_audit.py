"""Verify the documentation audit finds gaps without modifying source files."""

from robocub.tools.audit_documentation import audit, find_documentation_gaps


def test_audit_reports_missing_function_docstring(tmp_path):
    """A documented module with one bare function should produce one gap."""

    module = tmp_path / "sample.py"
    module.write_text('"""Example module."""\n\ndef undocumented():\n    return 1\n')

    gaps = find_documentation_gaps(module)

    assert [(gap.kind, gap.name) for gap in gaps] == [("function", "undocumented")]


def test_audit_accepts_fully_documented_module(tmp_path):
    """A module and function with intent docstrings should pass cleanly."""

    module = tmp_path / "sample.py"
    module.write_text(
        '"""Example module."""\n\ndef documented():\n'
        '    """Return the example value."""\n    return 1\n'
    )

    assert audit([module]) == []
