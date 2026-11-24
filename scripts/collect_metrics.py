"""Collect the measurable facts of this repository into reports/metrics.json.

Everything written here is measured by running the code in this repository;
no number is typed in by hand.

Usage:  python scripts/collect_metrics.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Files whose content changes with every run, so they must not enter the
# tree hash.
VOLATILE = {"reports/metrics.json"}


def _run(args, cwd=REPO_ROOT):
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _loc(path: Path) -> int:
    lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                lines += 1
    return lines


def _tree_hash() -> str:
    """Hash the committed content of the repository.

    The hash is built from the git object ids of the tracked files, so it does
    not depend on the line ending convention of the checkout. The same commit
    therefore produces the same tree hash on every machine.
    """
    listing = _run(["git", "ls-files", "-s"])
    if listing.returncode == 0 and listing.stdout.strip():
        digest = hashlib.sha256()
        for line in sorted(listing.stdout.splitlines()):
            parts = line.split()
            if len(parts) < 3:
                continue
            blob = parts[1]
            name = parts[3] if len(parts) > 3 else ""
            if name in VOLATILE:
                continue
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(blob.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    digest = hashlib.sha256()
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in VOLATILE:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".venv"} for part in path.parts):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_files():
    result = _run(["git", "ls-files"])
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def collect() -> dict:
    package_files = sorted((REPO_ROOT / "calib").glob("*.py"))
    test_files = sorted((REPO_ROOT / "tests").glob("*.py"))
    script_files = sorted((REPO_ROOT / "scripts").glob("*.py"))
    fixtures_dir = REPO_ROOT / "fixtures"

    pytest_run = _run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q"])
    pytest_text = pytest_run.stdout + pytest_run.stderr
    passed_match = re.search(r"(\d+)\s+passed", pytest_text)
    collected = _run([sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"])
    collected_match = re.search(r"(\d+)\s+tests?\s+collected", collected.stdout)
    if collected_match is None:
        collected_match = re.search(r"(\d+)\s+tests?", collected.stdout)

    cards_path = fixtures_dir / "demand_cards_500.jsonl"
    weights_path = fixtures_dir / "current_weights.json"
    cards = []
    if cards_path.exists():
        with cards_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cards.append(json.loads(line))
    candidates = [c for card in cards for c in card["candidates"]]
    distinct_items = {c["item_id"] for c in candidates}
    groups = sorted({card["group_id"] for card in cards})
    weights_payload = {}
    if weights_path.exists():
        weights_payload = json.loads(weights_path.read_text(encoding="utf-8"))

    run_files = sorted((REPO_ROOT / "calib_runs").glob("*.json"))
    runs = []
    for path in run_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": payload["run_id"],
                "seed": payload["seed"],
                "restarts": payload["restarts"],
                "evaluated_points": payload["evaluated_points"],
                "feasible_points": payload["feasible_points"],
                "rejected_points": payload["rejected_points"],
                "top1_weights": payload["top5"][0]["weights"] if payload["top5"] else None,
                "top1_ndcg": payload["top5"][0]["ndcg"] if payload["top5"] else None,
                "best_rejected_weights": (
                    payload["best_rejected"]["weights"] if payload["best_rejected"] else None
                ),
                "best_rejected_reason": (
                    payload["best_rejected"]["rejected_reason"] if payload["best_rejected"] else None
                ),
                "cv_ndcg_mean": payload["cv"]["ndcg_mean"],
                "cv_ndcg_std": payload["cv"]["ndcg_std"],
                "online_weights_written": payload["online_weights_written"],
            }
        )

    tracked = _git_files()
    metrics = {
        "python_version": platform.python_version(),
        "package_py_files": len(package_files),
        "package_loc": sum(_loc(p) for p in package_files),
        "test_files": len(test_files),
        "test_loc": sum(_loc(p) for p in test_files),
        "script_files": len(script_files),
        "script_loc": sum(_loc(p) for p in script_files),
        "test_cases": int(passed_match.group(1)) if passed_match else 0,
        "collected_tests": int(collected_match.group(1)) if collected_match else 0,
        "pytest_exit_code": pytest_run.returncode,
        "fixture_cards": len(cards),
        "fixture_candidates": len(candidates),
        "fixture_groups": len(groups),
        "fixture_distinct_items": len(distinct_items),
        "fixture_current_weights": weights_payload.get("weights"),
        "calib_runs": runs,
        "tracked_files": len(tracked),
        "tree_sha256": _tree_hash(),
        "online_weights_written": False,
    }
    return metrics


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = collect()
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for key in (
        "python_version",
        "package_py_files",
        "package_loc",
        "test_files",
        "test_cases",
        "collected_tests",
        "pytest_exit_code",
        "fixture_cards",
        "fixture_candidates",
        "fixture_groups",
        "fixture_distinct_items",
        "tracked_files",
        "tree_sha256",
    ):
        print("{0} {1}".format(key, metrics[key]))
    for run in metrics["calib_runs"]:
        print(
            "run {0} seed {1} restarts {2} evaluated_points {3} top1 {4} ndcg {5}".format(
                run["run_id"],
                run["seed"],
                run["restarts"],
                run["evaluated_points"],
                run["top1_weights"],
                run["top1_ndcg"],
            )
        )
    print("metrics", METRICS_PATH.relative_to(REPO_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
