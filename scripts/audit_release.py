"""Audit the Markdown-first release before packaging."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {".gitignore", ".github"}
FORBIDDEN_SUFFIXES = {".pdf", ".tex", ".bib", ".bbl", ".aux", ".log", ".html"}
MARKDOWN_LINK = re.compile(r"(?:src=|\]\()\"?([^\"\)]+)")


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            failures.append(f"forbidden path: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden file type: {relative}")

    required = [
        "README.md", "WHITEPAPER.md", "TECHNICAL_PAPER.md",
        "ARCHITECTURE.md", "REPRODUCIBILITY.md", "IMPLEMENTATION_GUIDE.md",
        "experiments/run_all.py", "experiments/run_million_agent.py",
        "results/reference/million_agent/million_agent_raw.csv",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            failures.append(f"missing required file: {name}")

    for document in ROOT.rglob("*.md"):
        content = document.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(content):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].strip()
            if not target or "{" in target or "$" in target:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"broken local link in {document.relative_to(ROOT)}: {target}")

    if failures:
        raise SystemExit("Release audit failed:\n" + "\n".join(failures))

    count = sum(1 for path in ROOT.rglob("*") if path.is_file())
    digest = hashlib.sha256((ROOT / "WHITEPAPER.md").read_bytes()).hexdigest()
    print(f"Release audit passed: {count} files; WHITEPAPER sha256={digest}")


if __name__ == "__main__":
    main()
