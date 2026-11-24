"""AC1: every reported candidate obeys the four constraints."""

from __future__ import annotations

import pytest

from calib import constraints as gate
from calib import space
from calib.data import load_current_weights, load_dataset
from calib.search import Evaluator, search_weights


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def current_weights():
    return load_current_weights()


@pytest.fixture(scope="module")
def run(dataset, current_weights):
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    result = search_weights(
        evaluator, current_weights, thresholds, seed=20261121, restarts=20
    )
    return current, thresholds, result


def test_top5_is_not_empty(run):
    _, _, result = run
    assert len(result["top5"]) > 0


def test_every_top5_candidate_passes_all_four_constraints(run, current_weights):
    current, thresholds, result = run
    for entry in result["top5"]:
        metrics = {
            "coverage": entry["coverage"],
            "gini": entry["gini"],
            "new_share": entry["new_share"],
        }
        passed, violations = gate.evaluate(
            entry["weights"], current_weights, metrics, thresholds
        )
        assert passed, (entry["weights"], violations)
        assert violations == []


def test_top5_is_sorted_by_ndcg_descending(run):
    _, _, result = run
    values = [entry["ndcg"] for entry in result["top5"]]
    assert values == sorted(values, reverse=True)


def test_top5_entries_are_inside_the_declared_weight_space(run, current_weights):
    _, _, result = run
    legal = set(space.enumerate_points(len(current_weights)))
    for entry in result["top5"]:
        assert tuple(entry["point"]) in legal
        assert abs(sum(entry["weights"]) - 1.0) < 1e-9
        for weight in entry["weights"]:
            assert 0.0 <= weight <= 0.5


def test_top5_never_contains_the_rejected_solution(run):
    _, _, result = run
    rejected = result["best_rejected"]
    if rejected is None:
        return
    top5 = {tuple(entry["weights"]) for entry in result["top5"]}
    assert tuple(rejected["weights"]) not in top5


def test_item_delta_constraint_is_respected(run, current_weights):
    current, thresholds, result = run
    for entry in result["top5"]:
        assert entry["max_delta"] <= thresholds["item_delta_max"] + 1e-12
