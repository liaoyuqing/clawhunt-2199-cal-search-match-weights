"""Scan the working tree and the git history for credential patterns.

The patterns are assembled from fragments so that this file does not match
itself. The scan is read only: it never prints the matched secret.

Exit codes
----------
0   scan executed and found nothing (history scan ran inside a git work tree)
1   at least one pattern matched in the working tree or in the git history
2   the history scan could not run because the directory is not a git work
    tree; the tree scan result alone must not be reported as a clean pass

The `history_scanned` line is always printed so that a caller can tell a
genuine clean history apart from a scan that never happened.

Usage:  python scripts/secret_scan.py
"""

from __future__ import annotations

import re
import subprocess
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

EXIT_CLEAN = 0
EXIT_HIT = 1
EXIT_NOT_GIT_WORK_TREE = 2


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


def scan_history():
    """Return (scanned, hits) for the git history of this repository.

    `scanned` is False when git could not read a history here; the caller
    must not treat that case as a clean result.
    """
    result = subprocess.run(
        ["git", "log", "-p", "--all"],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    if result.returncode != 0:
        return False, []
    if not result.stdout.strip():
        return False, []
    text = result.stdout.decode("utf-8", errors="replace")
    return True, [("history", name) for name in scan_text(text)]


def main() -> int:
    tree_hits = scan_tree()
    history_scanned, history_hits = scan_history()
    print("patterns", len(PATTERNS))
    print("tree_hits", len(tree_hits))
    for name, where in tree_hits:
        print("  tree_hit", name, where)
    print("history_hits", len(history_hits))
    for name, _ in history_hits:
        print("  history_hit", name)
    print("history_scanned", "yes" if history_scanned else "no")
    if tree_hits or history_hits:
        return EXIT_HIT
    if not history_scanned:
        print(
            "error: history scan did not run (not a git work tree); "
            "run this script inside the repository checkout"
        )
        return EXIT_NOT_GIT_WORK_TREE
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
