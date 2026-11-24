"""AC3: the output contains the best rejected solution and why it failed."""

from __future__ import annotations

from calib import constraints as gate
from calib.data import load_current_weights, load_dataset
from calib.search import Evaluator, random_restarts, search_weights


def _result():
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    return (
        search_weights(evaluator, current_weights, thresholds, seed=20261121, restarts=20),
        current_weights,
        thresholds,
    )


def test_best_rejected_is_present():
    result, _, _ = _result()
    assert result["best_rejected"] is not None


def test_best_rejected_reports_a_reason():
    result, _, _ = _result()
    rejected = result["best_rejected"]
    assert rejected["rejected_reason"]
    assert rejected["rejected_reason_text"]
    assert rejected["violations"]


def test_best_rejected_reason_follows_the_fixed_constraint_order():
    result, _, _ = _result()
    rejected = result["best_rejected"]
    order = list(gate.CONSTRAINT_NAMES)
    positions = [order.index(name) for name in rejected["rejected_reason"]]
    assert positions == sorted(positions)


def test_best_rejected_violates_exactly_the_reported_constraints():
    result, current_weights, thresholds = _result()
    rejected = result["best_rejected"]
    metrics = rejected["metrics"]
    passed, violations = gate.evaluate(
        rejected["weights"], current_weights, metrics, thresholds
    )
    assert passed is False
    assert [v["constraint"] for v in violations] == rejected["rejected_reason"]


def test_best_rejected_excess_total_matches_the_violations():
    result, _, _ = _result()
    rejected = result["best_rejected"]
    expected = sum(float(v["excess"]) for v in rejected["violations"])
    assert abs(rejected["excess_total"] - expected) < 1e-12


def test_best_rejected_is_the_highest_ndcg_rejected_point():
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    search = random_restarts(
        evaluator, current_weights, thresholds, seed=20261121, restarts=20
    )
    rejected = search["rejected"]
    assert rejected
    best_ndcg = max(float(record["ndcg"]) for record in rejected.values())
    result, _, _ = _result()
    assert abs(result["best_rejected"]["ndcg"] - best_ndcg) < 1e-12
