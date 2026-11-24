"""The four fairness / coverage metrics used by the constraint gate.

Every function is pure: it takes a list of exposure records (plain dicts) and
returns a float. Record fields used:

    has_supply       bool   the demand card had matching supply
    exposure_count   number the card was exposed in the ranked result
    is_new_member    bool   the card belongs to a new member
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _exposure(record: Dict) -> float:
    value = record.get("exposure_count", 0)
    if value is None:
        return 0.0
    return float(value)


def supply_coverage(records: Iterable[Dict]) -> float:
    """Share of records that had supply (has_supply == True)."""
    recs = list(records)
    if not recs:
        return 0.0
    covered = sum(1 for r in recs if bool(r.get("has_supply")))
    return covered / len(recs)


def exposure_gini(records: Iterable[Dict]) -> float:
    """Gini coefficient of the exposure counts (0.0 when total exposure is 0)."""
    recs = list(records)
    if not recs:
        return 0.0
    values = sorted(_exposure(r) for r in recs)
    n = len(values)
    total = sum(values)
    if total <= 0.0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    gini = (2.0 * weighted - (n + 1) * total) / (n * total)
    if gini < 0.0:
        return 0.0
    if gini > 1.0:
        return 1.0
    return gini


def new_member_exposure_share(records: Iterable[Dict]) -> float:
    """Share of total exposure that went to new members (is_new_member == True)."""
    recs = list(records)
    total = sum(_exposure(r) for r in recs)
    if total <= 0.0:
        return 0.0
    new_exposure = sum(_exposure(r) for r in recs if bool(r.get("is_new_member")))
    return new_exposure / total


def item_deltas(weights: Sequence[float], current_weights: Sequence[float]) -> List[float]:
    """Per-item absolute difference between candidate and current weights."""
    ws = [float(w) for w in weights]
    cw = [float(w) for w in current_weights]
    if len(ws) != len(cw):
        raise ValueError("weights and current_weights must have the same length")
    return [abs(a - b) for a, b in zip(ws, cw)]


def max_relative_delta(weights: Sequence[float], current_weights: Sequence[float]) -> float:
    """Largest per-item change versus the current weights.

    Weights are fractions of a unit total, so the absolute difference already
    is the relative change against the current weight vector.
    """
    deltas = item_deltas(weights, current_weights)
    if not deltas:
        return 0.0
    return max(deltas)


def compute_metrics(records: Iterable[Dict]) -> Dict[str, float]:
    """Convenience helper returning the three record-based metrics."""
    recs = list(records)
    return {
        "coverage": supply_coverage(recs),
        "gini": exposure_gini(recs),
        "new_share": new_member_exposure_share(recs),
    }
