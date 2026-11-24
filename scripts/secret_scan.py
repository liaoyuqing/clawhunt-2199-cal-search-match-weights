"""Scan the working tree and the git history for credential patterns.

The patterns are assembled from fragments so that this file does not match
itself. The scan is read only: it never prints the matched secret.

Usage:  python scripts/secret_scan.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}

PATTERNS = [
    ("agent_key", re.compile("c" + "ph_[0-9a-f]{32}")),
    ("github_oauth", re.compile("g" + "ho_[0-9A-Za-z]{36,}")),
    ("github_pat", re.compile("g" + "hp_[0-9A-Za-z]{36,}")),
    ("openai_key", re.compile("s" + "k-[0-9A-Za-z]{20,}")),
    ("aws_key", re.compile("A" + "KIA[0-9A-Z]{16}")),
    ("private_key", re.compile("-----BEGIN " + "[A-Z ]*PRIVATE KEY-----")),
]


def scan_text(text: str) -> list:
    hits = []
    for name, pattern in PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


def scan_tree() -> list:
    hits = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in scan_text(text):
            hits.append((name, path.relative_to(REPO_ROOT).as_posix()))
    return hits


def scan_history() -> list:
    result = subprocess.run(
        ["git", "log", "-p", "--all"],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    text = result.stdout.decode("utf-8", errors="replace")
    return [("history", name) for name in scan_text(text)]


def main() -> int:
    tree_hits = scan_tree()
    history_hits = scan_history()
    print("patterns", len(PATTERNS))
    print("tree_hits", len(tree_hits))
    for name, where in tree_hits:
        print("  tree_hit", name, where)
    print("history_hits", len(history_hits))
    for name, _ in history_hits:
        print("  history_hit", name)
    if tree_hits or history_hits:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
