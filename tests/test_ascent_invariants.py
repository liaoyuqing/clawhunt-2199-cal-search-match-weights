"""Invariants of the coordinate ascent: it may only return a feasible point.

The ascent is the only place where an infeasible point could leak into the
top 5, so these tests pin two properties:

  1. from every one of the 891 legal starts, the returned point satisfies all
     four constraints (a property of the real fixture);
  2. an infeasible point never becomes the incumbent, even when it is the
     highest scoring point the walk can see.

Property 2 needs a surface where an infeasible point really is the best one.
On the shipped fixture that situation never arises: the NDCG surface has a
single local maximum and it is feasible (see reports/ndcg_surface.md), so an
ascent that ignored the constraints would still end on a feasible point and
a test against the real data could not tell the two behaviours apart. The
second test therefore drives the ascent with a stub evaluator that puts the
global maximum on an infeasible weight vector.
"""

from __future__ import annotations

import pytest

from calib import constraints as gate
from calib import space
from calib.data import load_current_weights, load_dataset
from calib.search import Evaluator, coordinate_ascent


@pytest.fixture(scope="module")
def context():
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = gate.derive_thresholds(current)
    return evaluator, current_weights, thresholds


class StubEvaluator:
    """Evaluator whose highest scoring point is an infeasible weight vector.

    It mimics `calib.search.Evaluator.evaluate`: the same keys, and metrics
    that satisfy the coverage / gini / new_share thresholds for every weight
    vector, so the only constraint that can bite is the per item delta.
    """

    def __init__(self, champion, champion_ndcg=0.99, baseline_ndcg=0.50):
        self.champion = tuple(int(v) for v in champion)
        self.champion_ndcg = float(champion_ndcg)
        self.baseline_ndcg = float(baseline_ndcg)
        self.calls = []

    def evaluate(self, weights):
        point = space.point_from_weights(weights)
        self.calls.append(point)
        ndcg = self.champion_ndcg if point == self.champion else self.baseline_ndcg
        return {
            "ndcg_mean": ndcg,
            "coverage": 0.85,
            "gini": 0.70,
            "new_share": 0.26,
            "n_cards": 500,
            "n_slots": 5000,
        }


def _thresholds_for_the_stub():
    return gate.derive_thresholds(
        {"coverage": 0.84, "gini": 0.7235, "new_share": 0.2474}
    )


def _is_feasible(weights, current_weights, thresholds, evaluator):
    return gate.evaluate(weights, current_weights, evaluator.evaluate(weights), thresholds)


def test_every_start_ends_on_a_feasible_point(context):
    """All 891 legal starts, one ascent each, must end on a feasible point."""
    evaluator, current_weights, thresholds = context
    points = space.enumerate_points(len(current_weights))
    assert len(points) == 891
    infeasible_starts = 0
    for start in points:
        start_weights = space.to_weights(start)
        start_passed, _ = _is_feasible(
            start_weights, current_weights, thresholds, evaluator
        )
        if not start_passed:
            infeasible_starts += 1
        result = coordinate_ascent(
            evaluator, start, current_weights, thresholds, max_rounds=50
        )
        weights = result["best_weights"]
        passed, violations = _is_feasible(weights, current_weights, thresholds, evaluator)
        assert passed, (start, result["best_point"], violations)
        assert result["best_feasible"] is True
    # The grid really does contain infeasible points, so the loop above is not
    # passing merely because every point happens to be feasible.
    assert infeasible_starts > 0


def test_an_infeasible_start_does_not_become_the_incumbent():
    """The best point in sight is infeasible, so the walk must cross away.

    The stub makes [0.5, 0.2, 0.15, 0.15] the global maximum of the objective,
    and that vector breaks the item_delta constraint (0.25 away from the
    current 0.25 on the first item, limit 0.2). An ascent that compared NDCG
    only would keep it; the required behaviour is to end on a feasible point
    and report a lower NDCG.
    """
    current_weights = [0.25, 0.25, 0.25, 0.25]
    thresholds = _thresholds_for_the_stub()
    start = space.point_from_weights([0.5, 0.2, 0.15, 0.15])
    stub = StubEvaluator(start)
    passed, violations = _is_feasible(start_weights := space.to_weights(start),
                                      current_weights, thresholds, stub)
    assert passed is False
    assert [v["constraint"] for v in violations] == ["item_delta"]
    result = coordinate_ascent(
        stub, start, current_weights, thresholds, max_rounds=50
    )
    assert result["best_feasible"] is True
    assert tuple(result["best_point"]) != start
    end_weights = result["best_weights"]
    end_passed, end_violations = _is_feasible(
        end_weights, current_weights, thresholds, stub
    )
    assert end_passed is True
    assert end_violations == []
    # The walk gave up 0.49 of NDCG rather than accept the infeasible point.
    assert result["best_ndcg"] < stub.champion_ndcg
    assert start_weights == space.to_weights(start)


def test_a_feasible_improvement_is_still_accepted():
    """Positive control: the guard must not freeze the walk at its start.

    Same stub, but the champion this time is a feasible vector, so the ascent
    has to reach it. Without this control, "never returns an infeasible point"
    could be satisfied by an ascent that never moves at all.
    """
    current_weights = [0.25, 0.25, 0.25, 0.25]
    thresholds = _thresholds_for_the_stub()
    # One unit transfer away from the start, which is what a single greedy
    # step can reach; two transfers away would be unreachable by construction.
    champion = space.point_from_weights([0.30, 0.20, 0.25, 0.25])
    start = space.point_from_weights([0.25, 0.25, 0.25, 0.25])
    stub = StubEvaluator(champion)
    champion_passed, _ = _is_feasible(
        space.to_weights(champion), current_weights, thresholds, stub
    )
    assert champion_passed is True
    result = coordinate_ascent(
        stub, start, current_weights, thresholds, max_rounds=50
    )
    assert tuple(result["best_point"]) == champion
    assert result["best_feasible"] is True
    assert abs(float(result["best_ndcg"]) - stub.champion_ndcg) < 1e-12
