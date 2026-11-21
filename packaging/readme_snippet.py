"""Prints the first fenced code block under a README heading, so CI runs the docs verbatim.

Usage: python packaging/readme_snippet.py "### macOS" [README.md]
"""
import sys
from pathlib import Path


def snippet(readme: str, heading: str) -> str:
    lines = readme.splitlines()
    start = lines.index(heading)
    fence = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("```"))
    end = next(i for i in range(fence + 1, len(lines)) if lines[i].startswith("```"))
    return "\n".join(lines[fence + 1 : end]) + "\n"


if __name__ == "__main__":
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "README.md"
    sys.stdout.write(snippet(path.read_text(encoding="utf-8"), sys.argv[1]))
