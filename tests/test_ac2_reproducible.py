"""AC2: the same seed reproduces exactly the same top 5 candidates."""

from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

import pytest

from calib import constraints as gate
from calib.data import load_current_weights, load_dataset
from calib.run import main as run_main
from calib.search import Evaluator, search_weights

REPO_ROOT = Path(__file__).resolve().parent.parent


def _search(seed: int):
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    result = search_weights(
        evaluator, current_weights, thresholds, seed=seed, restarts=20
    )
    return result


def test_same_seed_gives_the_same_top5_weights():
    first = _search(20261121)
    second = _search(20261121)
    assert [e["weights"] for e in first["top5"]] == [e["weights"] for e in second["top5"]]
    assert [e["ndcg"] for e in first["top5"]] == [e["ndcg"] for e in second["top5"]]


def test_same_seed_gives_the_same_rejected_solution():
    first = _search(20261121)["best_rejected"]
    second = _search(20261121)["best_rejected"]
    assert first == second


def test_two_cli_runs_write_byte_identical_files(tmp_path):
    out_dir = tmp_path / "runs"
    assert run_main(["--seed", "20261121", "--out-dir", str(out_dir)]) == 0
    first = (out_dir / "run_seed20261121_r20.json").read_bytes()
    assert run_main(["--seed", "20261121", "--out-dir", str(out_dir)]) == 0
    second = (out_dir / "run_seed20261121_r20.json").read_bytes()
    assert first == second
    payload = json.loads(first.decode("utf-8"))
    assert payload["seed"] == 20261121
    assert payload["online_weights_written"] is False


def test_cli_runs_do_not_touch_the_current_weights_snapshot(tmp_path):
    weights_file = REPO_ROOT / "fixtures" / "current_weights.json"
    before = weights_file.read_bytes()
    run_main(["--seed", "20261121", "--out-dir", str(tmp_path / "runs")])
    assert weights_file.read_bytes() == before
