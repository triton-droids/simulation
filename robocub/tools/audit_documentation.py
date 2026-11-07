"""Report missing Python documentation without changing project behavior.

This developer tool makes the documentation standard measurable. It scans the
requested folders with Python's syntax tree, reports missing module, class, and
function docstrings, and optionally returns a failing exit status for CI.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = ("App/Scripts", "robocub")
IGNORED_PARTS = {".cache", ".git", ".venv", "__pycache__"}


@dataclass(frozen=True)
class DocumentationGap:
    """One missing docstring at a precise source location."""

    path: Path
    line: int
    kind: str
    name: str

    def render(self) -> str:
        """Return a concise line suitable for terminals and CI logs."""

        return f"{self.path}:{self.line}: missing {self.kind} docstring for {self.name}"


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files below files or directories while skipping caches."""

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*.py")):
            if not any(part in IGNORED_PARTS for part in candidate.parts):
                yield candidate


def find_documentation_gaps(path: Path) -> list[DocumentationGap]:
    """Parse one module and return every missing module/class/function docstring.

    Syntax errors propagate to the caller because a file that cannot be parsed
    cannot be audited reliably and should not be silently ignored.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    gaps: list[DocumentationGap] = []
    display_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path

    if ast.get_docstring(tree) is None:
        gaps.append(DocumentationGap(display_path, 1, "module", path.stem))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is None:
            gaps.append(DocumentationGap(display_path, node.lineno, "class", node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
            gaps.append(DocumentationGap(display_path, node.lineno, "function", node.name))
    return gaps


def audit(paths: Iterable[Path]) -> list[DocumentationGap]:
    """Collect and sort documentation gaps across all selected Python files."""

    gaps = [gap for path in iter_python_files(paths) for gap in find_documentation_gaps(path)]
    return sorted(gaps, key=lambda gap: (str(gap.path), gap.line, gap.kind, gap.name))


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for local and CI documentation checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    parser.add_argument("--strict", action="store_true", help="Exit with status 1 when gaps exist.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit, print each gap, and return the requested status code."""

    arguments = build_parser().parse_args(argv)
    paths = [(PROJECT_ROOT / value).resolve() for value in arguments.paths]
    gaps = audit(paths)
    for gap in gaps:
        print(gap.render())
    print(f"Documentation gaps: {len(gaps)}")
    return 1 if arguments.strict and gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
