"""Weight search space for the search-match-weight calibration.

Representation
--------------
Weights are kept as integer "step units" so that no floating point error can
creep into the sum constraint:

    1 step unit            = 0.05  (the grid step)
    per item range         = 0 .. 0.5   ->  0 .. 10 step units
    total (equality)       = 1.0        ->  exactly 20 step units

Every count is produced by a rolling-array dynamic program; the grid is never
brute forced over 11**k combinations.

Reference counts (verified by tests):
    k = 3 : exact(sum == 20) = 66     , sum <= 20 -> 1111
    k = 4 : exact(sum == 20) = 891    , sum <= 20 -> 7766
    k = 5 : exact(sum == 20) = 7051   , sum <= 20 -> 43120
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[int, ...]

STEP: float = 0.05
CAP_UNITS: int = 10
TOTAL_UNITS: int = 20
BASIS_POINTS_PER_UNIT: int = 500  # 1 step unit == 0.05 == 500 basis points


def enumerate_points(k: int, cap: int = CAP_UNITS, total: int = TOTAL_UNITS) -> List[Point]:
    """Return every legal weight point for k items.

    A point is a tuple of k integers in [0, cap] whose sum is exactly total.
    The returned list is sorted (lexicographic) so callers get a deterministic
    order without relying on dict / set iteration order.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if cap < 0 or total < 0:
        raise ValueError("cap and total must be non-negative")

    points: List[Point] = []

    def rec(prefix: Point, remaining: int, idx: int) -> None:
        if idx == k:
            if remaining == 0:
                points.append(prefix)
            return
        left = k - idx - 1
        lo = max(0, remaining - left * cap)
        hi = min(cap, remaining)
        for value in range(lo, hi + 1):
            rec(prefix + (value,), remaining - value, idx + 1)

    rec((), total, 0)
    points.sort()
    return points


def _dp_sum_counts(k: int, cap: int, total: int) -> List[int]:
    """Return counts[d] = number of k-tuples in [0, cap]**k with sum exactly d.

    Rolling-array DP over the dimensions; O(k * total * cap) time, O(total) space.
    """
    dp = [0] * (total + 1)
    dp[0] = 1
    for _ in range(k):
        nxt = [0] * (total + 1)
        for s in range(total + 1):
            acc = 0
            hi = min(cap, s)
            for v in range(0, hi + 1):
                acc += dp[s - v]
            nxt[s] = acc
        dp = nxt
    return dp


def count_points_exact(k: int, cap: int = CAP_UNITS, total: int = TOTAL_UNITS) -> int:
    """Number of legal points whose total is exactly total step units."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if cap < 0 or total < 0:
        raise ValueError("cap and total must be non-negative")
    return _dp_sum_counts(k, cap, total)[total]


def count_points_le(k: int, cap: int = CAP_UNITS, total: int = TOTAL_UNITS) -> int:
    """Number of legal points whose total is at most total step units."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if cap < 0 or total < 0:
        raise ValueError("cap and total must be non-negative")
    return sum(_dp_sum_counts(k, cap, total))


def to_weights(point: Sequence[int]) -> Tuple[float, ...]:
    """Convert a step-unit point into float weights (point * 0.05)."""
    return tuple(round(int(u) * STEP, 10) for u in point)


def to_basis_points(point: Sequence[int]) -> Tuple[int, ...]:
    """Convert a step-unit point into basis points (1.0 == 10000 bp)."""
    return tuple(int(u) * BASIS_POINTS_PER_UNIT for u in point)


def point_from_weights(weights: Sequence[float]) -> Point:
    """Inverse of to_weights: snap float weights back onto the integer grid."""
    units = []
    for w in weights:
        value = round(float(w) / STEP)
        if abs(value * STEP - float(w)) > 1e-9:
            raise ValueError("weight %r is not on the 0.05 grid" % (w,))
        units.append(int(value))
    return tuple(units)


def step_units(weights: Sequence[float]) -> int:
    """Total of float weights expressed in step units (integer, exact)."""
    return sum(point_from_weights(weights))


if __name__ == "__main__":
    for k in (3, 4, 5):
        exact = count_points_exact(k)
        at_most = count_points_le(k)
        print("k=%d exact(sum==20)=%d  sum<=20 -> %d" % (k, exact, at_most))
