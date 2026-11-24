"""Tests for the grouped cross validation and the constraint gate."""

from __future__ import annotations

import numpy as np

from calib import constraints as gate
from calib.cv import cross_validate, group_folds
from calib.data import group_ids, load_current_weights, load_dataset


def test_folds_are_disjoint_and_cover_every_group():
    dataset = load_dataset()
    folds = group_folds(dataset, 5)
    assert len(folds) == 5
    flat = [group for fold in folds for group in fold]
    assert sorted(flat) == group_ids(dataset)
    assert len(set(flat)) == len(flat)


def test_every_fold_holds_two_groups_and_one_hundred_cards():
    dataset = load_dataset()
    for fold in group_folds(dataset, 5):
        assert len(fold) == 2
        cards = [c for c in dataset if c["group_id"] in set(fold)]
        assert len(cards) == 100


def test_cross_validation_reports_mean_and_population_std():
    dataset = load_dataset()
    weights = load_current_weights()
    cv = cross_validate(dataset, weights, n_splits=5, k=10)
    assert len(cv["folds"]) == 5
    values = [row["ndcg"] for row in cv["folds"]]
    assert abs(cv["ndcg_mean"] - float(np.mean(values))) < 1e-12
    assert abs(cv["ndcg_std"] - float(np.std(values, ddof=0))) < 1e-12


def test_cross_validation_is_reproducible():
    dataset = load_dataset()
    weights = load_current_weights()
    first = cross_validate(dataset, weights, n_splits=5, k=10)
    second = cross_validate(dataset, weights, n_splits=5, k=10)
    assert first == second


def test_thresholds_are_derived_from_the_current_metrics():
    thresholds = gate.derive_thresholds(
        {"coverage": 0.80, "gini": 0.70, "new_share": 0.25}
    )
    assert abs(thresholds["coverage_min"] - 0.75) < 1e-12
    assert abs(thresholds["gini_max"] - 0.75) < 1e-12
    assert abs(thresholds["new_share_min"] - 0.25) < 1e-12
    assert abs(thresholds["item_delta_max"] - 0.2) < 1e-12


def test_gate_reports_the_breached_constraint_with_the_excess():
    current = {"coverage": 0.80, "gini": 0.70, "new_share": 0.25}
    thresholds = gate.derive_thresholds(current)
    metrics = {"coverage": 0.70, "gini": 0.70, "new_share": 0.25}
    passed, violations = gate.evaluate(
        [0.5, 0.2, 0.15, 0.15], [0.25, 0.25, 0.25, 0.25], metrics, thresholds
    )
    assert passed is False
    names = [v["constraint"] for v in violations]
    assert "coverage" in names
    assert "item_delta" in names
    for violation in violations:
        assert violation["excess"] > 0.0


def test_rejected_reason_uses_the_fixed_order():
    violations = [
        {"constraint": "item_delta", "value": 0.3, "limit": 0.2, "excess": 0.1},
        {"constraint": "coverage", "value": 0.7, "limit": 0.75, "excess": 0.05},
    ]
    assert gate.rejected_reason(violations) == ["coverage", "item_delta"]
