"""Unit tests for the weight space, the NDCG formula and the metrics."""

from __future__ import annotations

from calib import metrics as metric_tools
from calib import ndcg
from calib import space


def test_search_space_counts_match_the_reference_numbers():
    assert space.count_points_exact(3) == 66
    assert space.count_points_exact(4) == 891
    assert space.count_points_exact(5) == 7051
    assert space.count_points_le(3) == 1111
    assert space.count_points_le(4) == 7766
    assert space.count_points_le(5) == 43120


def test_every_enumerated_point_is_legal():
    points = space.enumerate_points(4)
    assert len(points) == 891
    for point in points:
        assert len(point) == 4
        assert sum(point) == 20
        assert all(0 <= value <= 10 for value in point)


def test_enumerated_points_are_sorted_and_unique():
    points = space.enumerate_points(4)
    assert points == sorted(points)
    assert len(set(points)) == len(points)


def test_weights_round_trip_through_the_integer_grid():
    for point in space.enumerate_points(4)[::37]:
        weights = space.to_weights(point)
        assert abs(sum(weights) - 1.0) < 1e-9
        assert space.point_from_weights(weights) == point
        assert space.step_units(weights) == 20


def test_off_grid_weights_are_rejected():
    try:
        space.point_from_weights([0.25, 0.25, 0.25, 0.24])
    except ValueError:
        return
    raise AssertionError("off grid weights must raise ValueError")


def test_ndcg_of_a_perfect_ranking_is_one():
    assert abs(ndcg.ndcg_at_k([3, 2, 1], [3, 2, 1], 10) - 1.0) < 1e-12


def test_ndcg_of_the_reversed_ranking_matches_the_hand_value():
    expected = 0.6806060568
    actual = ndcg.ndcg_at_k([1, 2, 3], [3, 2, 1], 10)
    assert abs(actual - expected) < 1e-9


def test_ndcg_of_an_empty_ideal_list_is_zero():
    assert ndcg.ndcg_at_k([], [], 10) == 0.0


def test_dcg_matches_the_hand_computed_value():
    assert abs(ndcg.dcg_at_k([3, 2, 1], 10) - 9.3927892607) < 1e-9


def test_supply_coverage_counts_the_cards_with_supply():
    records = [{"has_supply": True}, {"has_supply": False}, {"has_supply": True}]
    assert abs(metric_tools.supply_coverage(records) - 2.0 / 3.0) < 1e-12


def test_gini_is_zero_for_a_flat_exposure_distribution():
    records = [{"exposure_count": 3}, {"exposure_count": 3}, {"exposure_count": 3}]
    assert abs(metric_tools.exposure_gini(records)) < 1e-12


def test_new_member_share_uses_the_exposure_slots():
    records = [
        {"exposure_count": 2, "is_new_member": True},
        {"exposure_count": 2, "is_new_member": False},
    ]
    assert abs(metric_tools.new_member_exposure_share(records) - 0.5) < 1e-12


def test_max_relative_delta_is_the_largest_weight_move():
    assert abs(metric_tools.max_relative_delta([0.4, 0.2, 0.2, 0.2], [0.25] * 4) - 0.15) < 1e-12
