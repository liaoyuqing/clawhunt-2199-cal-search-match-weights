"""Constrained weight search for the search-match-weight calibration.

The search space is the integer step-unit grid from `calib.space`:
four weights, each 0..0.5 in steps of 0.05, summing to exactly 1.0.
Because the total is an equality constraint, a coordinate move is a
transfer of one unit (0.05) from one weight to another.

Each candidate point is scored with NDCG@10 and gated by the four
constraints in `calib.constraints`:

    coverage   >= current_coverage - 0.05
    gini       <= current_gini     + 0.05
    new_share  >= current_new_share
    item_delta <= 0.2

Only feasible points may enter the top 5 candidates. Every infeasible
point that was evaluated is kept, and the one with the highest NDCG
becomes `best_rejected` together with the reason it was rejected.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import constraints as constraint_gate
from . import space

Point = Tuple[int, ...]

_EPS = 1e-12


class Evaluator:
    """Vectorised evaluation of weight vectors on a fixed dataset."""

    def __init__(self, dataset: List[Dict], k: int = 10) -> None:
        self.k = int(k)
        n_cards = len(dataset)
        counts = [len(card.get("candidates") or []) for card in dataset]
        width = max(counts) if counts else 0
        dim = 0
        for card in dataset:
            matrix = card.get("features_matrix")
            if matrix is None:
                matrix = np.asarray(
                    [c["features"] for c in card.get("candidates", [])], dtype=float
                )
            if len(matrix):
                dim = int(matrix.shape[1])
                break
        if dim == 0:
            dim = 4
        features = np.zeros((n_cards, width, dim), dtype=float)
        ids = np.zeros((n_cards, width), dtype="U32")
        supply = np.zeros((n_cards, width), dtype=bool)
        new_member = np.zeros((n_cards, width), dtype=bool)
        rel = np.zeros((n_cards, width), dtype=float)
        unique_ids = sorted(
            {str(c["item_id"]) for card in dataset for c in (card.get("candidates") or [])}
        )
        lookup = {item_id: i for i, item_id in enumerate(unique_ids)}
        item_index = np.full((n_cards, width), -1, dtype=np.int64)
        for row, card in enumerate(dataset):
            cands = card.get("candidates") or []
            if not cands:
                continue
            matrix = card.get("features_matrix")
            if matrix is None:
                matrix = np.asarray([c["features"] for c in cands], dtype=float)
            for col, cand in enumerate(cands[:width]):
                features[row, col] = matrix[col]
                ids[row, col] = str(cand["item_id"])
                supply[row, col] = bool(cand.get("has_supply"))
                new_member[row, col] = bool(cand.get("is_new_member"))
                rel[row, col] = float(cand.get("rel", 0))
                item_index[row, col] = lookup[str(cand["item_id"])]
        self.n_cards = n_cards
        self.width = width
        self.dim = dim
        self.features = features
        self.ids = ids
        self.supply = supply
        self.new_member = new_member
        self.rel = rel
        self.item_index = item_index
        self.counts = np.asarray(counts, dtype=np.int64)
        self.n_items = len(unique_ids)
        self.item_ids = unique_ids
        self.k_eff = max(0, min(self.k, width))
        self.discount = 1.0 / np.log2(np.arange(self.k_eff, dtype=float) + 2.0)
        ideal = np.zeros(n_cards, dtype=float)
        for row, card in enumerate(dataset):
            rels = sorted(
                (float(c.get("rel", 0)) for c in (card.get("candidates") or [])), reverse=True
            )
            gains = np.power(2.0, np.asarray(rels, dtype=float)[: self.k_eff]) - 1.0
            disc = 1.0 / np.log2(np.arange(len(gains), dtype=float) + 2.0)
            ideal[row] = float((gains * disc).sum())
        self.idcg = ideal
        self._cache: Dict[Tuple[float, ...], Dict[str, float]] = {}

    def evaluate(self, weights: Sequence[float]) -> Dict[str, float]:
        """Metrics of one weight vector: ndcg_mean / coverage / gini / new_share."""
        key = tuple(round(float(w), 10) for w in weights)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vector = np.asarray(key, dtype=float)
        scores = self.features @ vector
        invalid = np.arange(self.width)[None, :] >= self.counts[:, None]
        scores = np.where(invalid, -np.inf, scores)
        order = np.lexsort((self.ids, -np.round(scores, 12)), axis=-1)
        selected = order[:, : self.k_eff]
        valid = np.arange(self.k_eff)[None, :] < self.counts[:, None]
        rel_sel = np.take_along_axis(self.rel, selected, axis=1)
        supply_sel = np.take_along_axis(self.supply, selected, axis=1)
        new_sel = np.take_along_axis(self.new_member, selected, axis=1)
        item_sel = np.take_along_axis(self.item_index, selected, axis=1)
        gains = np.power(2.0, rel_sel) - 1.0
        dcg = (gains * self.discount[None, :]).sum(axis=1)
        positive = self.idcg > 0.0
        denom = np.where(positive, self.idcg, 1.0)
        ndcg = np.where(positive, dcg / denom, 0.0)
        ndcg_mean = float(ndcg.mean()) if self.n_cards else 0.0
        coverage = float((supply_sel & valid).any(axis=1).mean()) if self.n_cards else 0.0
        n_slots = int(valid.sum())
        new_share = float(new_sel[valid].sum() / n_slots) if n_slots else 0.0
        if n_slots:
            counts = np.bincount(item_sel[valid].astype(np.int64), minlength=self.n_items)
        else:
            counts = np.zeros(self.n_items, dtype=np.int64)
        result = {
            "ndcg_mean": ndcg_mean,
            "coverage": coverage,
            "gini": self._gini(counts),
            "new_share": new_share,
            "n_cards": int(self.n_cards),
            "n_slots": n_slots,
        }
        self._cache[key] = result
        return result

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        """Gini coefficient of the per item exposure counts."""
        values = np.sort(np.asarray(counts, dtype=float))
        n = int(values.size)
        if n == 0:
            return 0.0
        total = float(values.sum())
        if total <= 0.0:
            return 0.0
        weighted = float(np.dot(np.arange(1, n + 1, dtype=float), values))
        gini = (2.0 * weighted - (n + 1) * total) / (n * total)
        return float(min(1.0, max(0.0, gini)))


def _neighbours(point: Point, step: int, cap: int) -> List[Point]:
    """One unit moved from one coordinate to another (sum is preserved)."""
    result: List[Point] = []
    size = len(point)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            candidate = list(point)
            candidate[i] += step
            candidate[j] -= step
            if 0 <= candidate[i] <= cap and 0 <= candidate[j] <= cap:
                result.append(tuple(candidate))
    return result


def _bfs_bridge(consider, start_point: Point, step: int, cap: int, max_bridge: int):
    """Find the best feasible point within max_bridge unit moves of start_point.

    An infeasible start with no feasible immediate neighbour still has to be
    able to reach the feasible region: a breadth first walk crosses the
    infeasible ring and returns the feasible point with the highest NDCG at the
    shallowest distance where any feasible point appears. Every visited point
    is recorded through `consider`, so the crossing points feed best_rejected.
    """
    seen = {start_point}
    frontier = [start_point]
    for _ in range(max(0, int(max_bridge))):
        next_frontier = []
        winners = []
        for node in frontier:
            for candidate in sorted(_neighbours(node, step, cap)):
                if candidate in seen:
                    continue
                seen.add(candidate)
                record = consider(candidate)
                if record["passed"]:
                    winners.append(record)
                next_frontier.append(candidate)
        if winners:
            winners.sort(key=lambda item: (-float(item["ndcg"]), tuple(item["point"])))
            return winners[0]
        frontier = next_frontier
        if not frontier:
            break
    return None


def coordinate_ascent(
    evaluator: Evaluator,
    start_point: Point,
    current_weights: Sequence[float],
    thresholds: Dict[str, float],
    max_rounds: int = 50,
    step: int = 1,
    cap: int = space.CAP_UNITS,
    max_bridge: int = 6,
) -> Dict[str, object]:
    """Coordinate ascent on the NDCG objective, recording every point.

    The incumbent is always a feasible point: an infeasible start is bridged to
    the nearest feasible region first, and an infeasible neighbour is never
    accepted as an improvement.
    """
    feasible: Dict[Point, Dict[str, object]] = {}
    rejected: Dict[Point, Dict[str, object]] = {}

    def consider(point: Point) -> Dict[str, object]:
        weights = space.to_weights(point)
        metrics = evaluator.evaluate(weights)
        passed, violations = constraint_gate.evaluate(
            weights, current_weights, metrics, thresholds
        )
        record: Dict[str, object] = {
            "point": point,
            "weights": list(weights),
            "ndcg": float(metrics["ndcg_mean"]),
            "metrics": metrics,
            "passed": bool(passed),
            "violations": violations,
        }
        bucket = feasible if passed else rejected
        previous = bucket.get(point)
        if previous is None or record["ndcg"] > previous["ndcg"] + _EPS:
            bucket[point] = record
        return record

    position = tuple(int(v) for v in start_point)
    start_record = consider(position)
    # The incumbent of the ascent must be feasible. An infeasible start is kept
    # in the rejected bucket as a best_rejected candidate, but it never becomes
    # the incumbent: the walk first crosses to a feasible point.
    best = None
    if start_record["passed"]:
        best = start_record
    bridge = None
    if best is None:
        bridge = _bfs_bridge(consider, position, step, cap, max_bridge)
        if bridge is not None:
            best = bridge
    history = [
        {
            "round": 0,
            "point": list(position),
            "weights": start_record["weights"],
            "ndcg": start_record["ndcg"],
            "feasible": start_record["passed"],
        }
    ]
    if bridge is not None:
        history.append(
            {
                "round": 0,
                "point": list(bridge["point"]),
                "weights": bridge["weights"],
                "ndcg": bridge["ndcg"],
                "feasible": True,
                "bridged_from": list(position),
            }
        )
    for round_index in range(1, int(max_rounds) + 1):
        if best is None:
            break
        anchor = best["point"]
        incumbent = float(best["ndcg"])
        improved = False
        for candidate in sorted(_neighbours(anchor, step, cap)):
            record = consider(candidate)
            # Only a point that satisfies all four constraints may become the
            # incumbent, so an infeasible point can never be reported as the
            # result of the ascent.
            if not record["passed"]:
                continue
            if record["ndcg"] > incumbent + _EPS:
                best = record
                incumbent = float(record["ndcg"])
                improved = True
        history.append(
            {
                "round": round_index,
                "point": list(best["point"]),
                "weights": best["weights"],
                "ndcg": best["ndcg"],
                "feasible": bool(best["passed"]),
            }
        )
        if not improved:
            break
    if best is None:
        # No feasible point was reachable from this start within max_bridge
        # moves; the start only contributes to the rejected pool.
        best = start_record
    return {
        "best_point": best["point"],
        "best_ndcg": best["ndcg"],
        "best_weights": best["weights"],
        "best_feasible": bool(best["passed"]),
        "history": history,
        "feasible": feasible,
        "rejected": rejected,
        "evaluated_points": len(feasible) + len(rejected),
    }


def random_restarts(
    evaluator: Evaluator,
    current_weights: Sequence[float],
    thresholds: Dict[str, float],
    seed: int = 20261121,
    restarts: int = 20,
    max_rounds: int = 50,
    step: int = 1,
    include_current: bool = True,
) -> Dict[str, object]:
    """Coordinate ascent restarted from seeded random points."""
    k = len(current_weights)
    points = space.enumerate_points(k)
    rng = np.random.default_rng(seed)
    starts: List[Point] = []
    seen = set()
    attempts = 0
    while len(starts) < int(restarts) and attempts < int(restarts) * 50:
        attempts += 1
        index = int(rng.integers(0, len(points)))
        point = points[index]
        if point in seen:
            continue
        seen.add(point)
        starts.append(point)
    current_point = space.point_from_weights(current_weights)
    if include_current and current_point not in seen:
        starts.append(current_point)
    feasible: Dict[Point, Dict[str, object]] = {}
    rejected: Dict[Point, Dict[str, object]] = {}
    summaries: List[Dict[str, object]] = []
    for index, start in enumerate(starts):
        result = coordinate_ascent(
            evaluator,
            start,
            current_weights,
            thresholds,
            max_rounds=max_rounds,
            step=step,
        )
        for point, record in result["feasible"].items():
            previous = feasible.get(point)
            if previous is None or record["ndcg"] > previous["ndcg"] + _EPS:
                feasible[point] = record
        for point, record in result["rejected"].items():
            previous = rejected.get(point)
            if previous is None or record["ndcg"] > previous["ndcg"] + _EPS:
                rejected[point] = record
        summaries.append(
            {
                "index": index,
                "start_point": list(start),
                "start_weights": list(space.to_weights(start)),
                "end_point": list(result["best_point"]),
                "end_weights": list(result["best_weights"]),
                "ndcg": float(result["best_ndcg"]),
                "feasible": bool(result["best_feasible"]),
                "rounds": len(result["history"]),
                "evaluated_points": int(result["evaluated_points"]),
            }
        )
    return {
        "starts": [list(p) for p in starts],
        "summaries": summaries,
        "feasible": feasible,
        "rejected": rejected,
        "evaluated_points": len(feasible) + len(rejected),
        "feasible_points": len(feasible),
        "rejected_points": len(rejected),
    }


def _candidate_entry(record: Dict[str, object], current_weights: Sequence[float]) -> Dict[str, object]:
    weights = [float(w) for w in record["weights"]]
    metrics = record["metrics"]
    diff = [round(w - float(c), 10) for w, c in zip(weights, current_weights)]
    return {
        "point": list(record["point"]),
        "weights": weights,
        "ndcg": float(record["ndcg"]),
        "coverage": float(metrics["coverage"]),
        "gini": float(metrics["gini"]),
        "new_share": float(metrics["new_share"]),
        "max_delta": float(max(abs(d) for d in diff)),
        "diff_vs_current": diff,
    }


def search_weights(
    evaluator: Evaluator,
    current_weights: Sequence[float],
    thresholds: Dict[str, float],
    seed: int = 20261121,
    restarts: int = 20,
    max_rounds: int = 50,
    step: int = 1,
    include_current: bool = True,
) -> Dict[str, object]:
    """Search the best feasible weight vectors and the best rejected one."""
    current = [float(w) for w in current_weights]
    search = random_restarts(
        evaluator,
        current,
        thresholds,
        seed=seed,
        restarts=restarts,
        max_rounds=max_rounds,
        step=step,
        include_current=include_current,
    )
    feasible = search["feasible"]
    rejected = search["rejected"]
    ordered = sorted(
        feasible.values(),
        key=lambda item: (-float(item["ndcg"]), tuple(item["point"])),
    )
    top5 = [_candidate_entry(item, current) for item in ordered[:5]]
    best_rejected = None
    if rejected:
        winner = sorted(
            rejected.values(),
            key=lambda item: (-float(item["ndcg"]), tuple(item["point"])),
        )[0]
        violations = winner["violations"]
        reasons = constraint_gate.rejected_reason(violations)
        metrics = winner["metrics"]
        weights = [float(w) for w in winner["weights"]]
        excess_total = float(sum(float(v["excess"]) for v in violations))
        text_parts = []
        for violation in violations:
            text_parts.append(
                "{0} 实测 {1:.6f}，限值 {2:.6f}，超出 {3:.6f}".format(
                    violation["constraint"],
                    float(violation["value"]),
                    float(violation["limit"]),
                    float(violation["excess"]),
                )
            )
        best_rejected = {
            "point": list(winner["point"]),
            "weights": weights,
            "ndcg": float(winner["ndcg"]),
            "metrics": {
                "coverage": float(metrics["coverage"]),
                "gini": float(metrics["gini"]),
                "new_share": float(metrics["new_share"]),
                "max_delta": float(max(abs(w - c) for w, c in zip(weights, current))),
            },
            "rejected_reason": reasons,
            "rejected_reason_text": "违反约束：" + "；".join(text_parts),
            "violations": violations,
            "excess_total": excess_total,
        }
    top5_weights = {tuple(entry["weights"]) for entry in top5}
    if best_rejected is not None:
        assert tuple(best_rejected["weights"]) not in top5_weights
    return {
        "top5": top5,
        "best_rejected": best_rejected,
        "seed": int(seed),
        "restarts": len(search["summaries"]),
        "evaluated_points": int(search["evaluated_points"]),
        "feasible_points": int(search["feasible_points"]),
        "rejected_points": int(search["rejected_points"]),
        "restart_summaries": search["summaries"],
    }
