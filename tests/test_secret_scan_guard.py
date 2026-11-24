"""Regression tests for the secret scan guard.

The scan has to distinguish three outcomes: clean, hit, and did-not-run.
The dangerous failure mode is reporting "clean" when the history scan never
executed, so these tests pin the exit code and the `history_scanned` line for
both a real git work tree and a plain directory.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "secret_scan.py"


def _run(cwd: Path):
    return subprocess.run(
        [sys.executable, str(cwd / "scripts" / "secret_scan.py")],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_the_scan_reports_matched_patterns_and_a_clean_result_in_the_repository():
    """""In the real checkout the history scan runs and the tree is clean."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = result.stdout + result.stderr
    assert "patterns " in combined
    assert "tree_hits " in combined
    assert "history_scanned yes" in combined
    assert result.returncode == 0, combined


def test_the_history_scan_is_not_silently_skipped_outside_a_git_work_tree(tmp_path):
    """A directory without .git must fail loudly instead of claiming a pass.

    Guards the regression where `git log` failed, the exception was swallowed
    and the script exited 0 with `history_hits 0`, which reads exactly like a
    clean scan.
    """
    target = tmp_path / "no_git_here"
    (target / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, target / "scripts" / "secret_scan.py")
    (target / "notes.txt").write_text("ordinary text, nothing secret\n", encoding="utf-8")
    assert not (target / ".git").exists()
    result = _run(target)
    combined = result.stdout + result.stderr
    assert "history_scanned no" in combined, combined
    assert result.returncode == 2, combined


def test_a_planted_secret_is_reported_and_fails_the_scan(tmp_path):
    """The scanner still detects a real looking credential in the tree."""
    target = tmp_path / "planted"
    (target / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, target / "scripts" / "secret_scan.py")
    token = "c" + "ph_" + "0123456789abcdef0123456789abcdef"
    (target / "leaked.txt").write_text("key = " + token + "\n", encoding="utf-8")
    result = _run(target)
    combined = result.stdout + result.stderr
    assert "tree_hits 1" in combined, combined
    assert result.returncode == 1, combined
    assert token not in combined
