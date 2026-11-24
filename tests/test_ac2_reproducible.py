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
from calib.search import Evaluator, random_restarts, search_weights

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

def _point_set(seed: int):
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    search = random_restarts(
        evaluator, current_weights, thresholds, seed=seed, restarts=20
    )
    evaluated = set(search["feasible"]) | set(search["rejected"])
    return [entry["weights"] for entry in search_weights(
        evaluator, current_weights, thresholds, seed=seed, restarts=20
    )["top5"]], evaluated


def test_same_seed_evaluates_exactly_the_same_points():
    """Reproducibility has two halves: same answer, and same work done.

    Matching top 5 alone would also hold if the search blindly returned a
    constant. Requiring the evaluated point set to match as well proves the
    restart sequence is a deterministic function of the seed.
    """
    _, first = _point_set(20261121)
    _, second = _point_set(20261121)
    assert first == second
    assert len(first) > 20


def test_different_seeds_do_explore_different_points():
    """A different seed walks a different part of the grid.

    The NDCG surface on this fixture has a single local maximum (see
    reports/ndcg_surface.md), so both seeds land on the same top 5. What still
    differs is the set of points the ascent visited, which is the observable
    proof that the random restarts are actually random and seed controlled.
    """
    top_a, points_a = _point_set(20261121)
    top_b, points_b = _point_set(7)
    assert top_a == top_b
    assert points_a != points_b
    assert points_a | points_b != points_a
