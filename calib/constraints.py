"""Constraint gate for the candidate weight vectors.

Four constraints, any violation eliminates the candidate:

    coverage    >=  current_coverage - 0.05   (5 percentage points of tolerance)
    gini        <=  current_gini     + 0.05
    new_share   >=  current_new_share
    item_delta  <=  0.2   (max |w_i - w_i_current|)

The rejection order is fixed: coverage -> gini -> new_share -> item_delta.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import max_relative_delta

CONSTRAINT_NAMES: Tuple[str, ...] = ("coverage", "gini", "new_share", "item_delta")

COVERAGE_TOLERANCE: float = 0.05
GINI_TOLERANCE: float = 0.05
ITEM_DELTA_LIMIT: float = 0.2

Violation = Dict[str, object]


def derive_thresholds(current_metrics: Dict[str, float]) -> Dict[str, float]:
    """Threshold dict derived from the metrics of the current weights."""
    return {
        "coverage_min": float(current_metrics["coverage"]) - COVERAGE_TOLERANCE,
        "gini_max": float(current_metrics["gini"]) + GINI_TOLERANCE,
        "new_share_min": float(current_metrics["new_share"]),
        "item_delta_max": ITEM_DELTA_LIMIT,
    }


def evaluate(
    weights: Sequence[float],
    current_weights: Sequence[float],
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, List[Violation]]:
    """Evaluate the four constraints for one candidate weight vector.

    Returns (passed, violations). Every violation entry carries:
        constraint : name from CONSTRAINT_NAMES
        value      : measured value of the metric
        limit      : the threshold it was compared against
        excess     : amount by which the limit is breached (0.0 when inside)
    """
    if thresholds is None:
        current = metrics.get("current")
        if not isinstance(current, dict):
            raise ValueError("thresholds is required unless metrics carries a 'current' block")
        thresholds = derive_thresholds(current)

    coverage = float(metrics["coverage"])
    gini = float(metrics["gini"])
    new_share = float(metrics["new_share"])
    item_delta = float(max_relative_delta(weights, current_weights))

    checks = (
        ("coverage", coverage, float(thresholds["coverage_min"]), "min"),
        ("gini", gini, float(thresholds["gini_max"]), "max"),
        ("new_share", new_share, float(thresholds["new_share_min"]), "min"),
        ("item_delta", item_delta, float(thresholds["item_delta_max"]), "max"),
    )

    violations: List[Violation] = []
    for name, value, limit, kind in checks:
        if kind == "min":
            breach = value < limit
            excess = limit - value if breach else 0.0
        else:
            breach = value > limit
            excess = value - limit if breach else 0.0
        if breach:
            violations.append(
                {
                    "constraint": name,
                    "value": value,
                    "limit": limit,
                    "excess": excess,
                }
            )

    return (len(violations) == 0), violations


def rejected_reason(violations: Sequence[Violation]) -> List[str]:
    """Rejected constraint names, always in the fixed order coverage, gini,
    new_share, item_delta. Empty list when nothing was violated.
    """
    breached = {str(v["constraint"]) for v in violations}
    return [name for name in CONSTRAINT_NAMES if name in breached]
