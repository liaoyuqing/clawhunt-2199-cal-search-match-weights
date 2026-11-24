"""Scoring of candidate items under a weight vector.

For a weight vector w the candidates of a demand card are ordered by

    score = w . features

descending, ties broken by ascending `item_id` so the order is fully
deterministic. The first k (default 10) candidates form the exposure set of
that card.

The three fairness metrics reuse the single implementation in `calib.metrics`
so the constraint gate and the report can never drift apart.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .metrics import exposure_gini, new_member_exposure_share, supply_coverage
from .ndcg import ndcg_at_k

DEFAULT_K = 10
_SCORE_DECIMALS = 12


def _weight_vector(weights: Sequence[float]) -> np.ndarray:
    return np.asarray(weights, dtype=float).ravel()


def _card_features(card: Dict) -> np.ndarray:
    matrix = card.get("features_matrix")
    if matrix is None:
        matrix = np.asarray(
            [cand["features"] for cand in card.get("candidates", [])], dtype=float
        )
    return np.asarray(matrix, dtype=float)


def rank_candidates(card: Dict, weights: Sequence[float]) -> List[Dict]:
    """Return the candidates of one card ordered by score (desc), id (asc)."""
    candidates = list(card.get("candidates") or [])
    if not candidates:
        return []
    scores = _card_features(card) @ _weight_vector(weights)
    ids = np.asarray([str(cand["item_id"]) for cand in candidates], dtype="U32")
    order = np.lexsort((ids, -np.round(scores, _SCORE_DECIMALS)))
    return [candidates[int(position)] for position in order]


def ndcg_for_card(card: Dict, weights: Sequence[float], k: int = DEFAULT_K) -> float:
    """NDCG@k of one demand card under the given weight vector."""
    ranked = rank_candidates(card, weights)[:k]
    all_candidates = list(card.get("candidates") or [])
    if not ranked or not all_candidates:
        return 0.0
    ranked_rels = [float(cand["rel"]) for cand in ranked]
    ideal_rels = sorted((float(cand["rel"]) for cand in all_candidates), reverse=True)
    return ndcg_at_k(ranked_rels, ideal_rels, k)


def _build_batch(dataset: List[Dict]) -> Dict:
    """Pad the dataset into dense arrays for a vectorised evaluation."""
    n_cards = len(dataset)
    counts = [len(card.get("candidates") or []) for card in dataset]
    width = max(counts) if counts else 0
    dim = 0
    for card in dataset:
        if card.get("candidates"):
            dim = _card_features(card).shape[1]
            break
    if dim == 0:
        dim = 4

    features = np.zeros((n_cards, width, dim), dtype=float)
    ids = np.zeros((n_cards, width), dtype="U32")
    supply = np.zeros((n_cards, width), dtype=bool)
    new_member = np.zeros((n_cards, width), dtype=bool)
    rel = np.zeros((n_cards, width), dtype=float)
    item_index = np.full((n_cards, width), -1, dtype=np.int64)

    unique_ids = sorted(
        {str(cand["item_id"]) for card in dataset for cand in (card.get("candidates") or [])}
    )
    item_lookup = {item_id: position for position, item_id in enumerate(unique_ids)}

    for row, card in enumerate(dataset):
        candidates = card.get("candidates") or []
        if not candidates:
            continue
        matrix = _card_features(card)
        for column, cand in enumerate(candidates[:width]):
            features[row, column] = matrix[column]
            ids[row, column] = str(cand["item_id"])
            supply[row, column] = bool(cand.get("has_supply"))
            new_member[row, column] = bool(cand.get("is_new_member"))
            rel[row, column] = float(cand.get("rel", 0))
            item_index[row, column] = item_lookup[str(cand["item_id"])]

    return {
        "n_cards": n_cards,
        "width": width,
        "counts": counts,
        "features": features,
        "ids": ids,
        "supply": supply,
        "new_member": new_member,
        "rel": rel,
        "item_index": item_index,
        "n_items": len(unique_ids),
        "item_ids": unique_ids,
    }


def evaluate_weights(
    dataset: List[Dict],
    weights: Sequence[float],
    k: int = DEFAULT_K,
    include_counts: bool = False,
) -> Dict[str, object]:
    """Evaluate one weight vector on the whole dataset.

    Returns a dict with:
        ndcg_mean        mean NDCG@10 over all demand cards
        coverage         share of cards whose exposure set contains a supplied item
        gini             gini coefficient of the item exposure counts
        new_share        share of exposure slots held by new members
        n_cards          number of demand cards
        n_slots          total number of exposure slots
        exposure_counts  item_id -> exposure count (only when include_counts)
    """
    batch = _build_batch(dataset)
    n_cards = batch["n_cards"]
    if n_cards == 0:
        empty = {
            "ndcg_mean": 0.0,
            "coverage": 0.0,
            "gini": 0.0,
            "new_share": 0.0,
            "n_cards": 0,
            "n_slots": 0,
        }
        if include_counts:
            empty["exposure_counts"] = {}
        return empty

    width = batch["width"]
    scores = batch["features"] @ _weight_vector(weights)
    invalid = np.arange(width)[None, :] >= np.asarray(batch["counts"])[:, None]
    scores = np.where(invalid, -np.inf, scores)
    order = np.lexsort((batch["ids"], -np.round(scores, _SCORE_DECIMALS)), axis=-1)

    k_eff = max(0, min(int(k), width))
    selected = order[:, :k_eff]
    valid_mask = np.arange(k_eff)[None, :] < np.asarray(batch["counts"])[:, None]

    item_of_slot = np.take_along_axis(batch["item_index"], selected, axis=1)
    supply_of_slot = np.take_along_axis(batch["supply"], selected, axis=1)
    new_of_slot = np.take_along_axis(batch["new_member"], selected, axis=1)
    rel_of_slot = np.take_along_axis(batch["rel"], selected, axis=1)

    # coverage: one record per card, has_supply = the card exposed a supplied item
    card_records = [
        {"has_supply": bool(supply_of_slot[row][valid_mask[row]].any())}
        for row in range(n_cards)
    ]
    coverage = supply_coverage(card_records)

    # new member exposure share: one record per exposure slot
    slot_records = []
    for row in range(n_cards):
        for column in range(k_eff):
            if not valid_mask[row][column]:
                continue
            slot_records.append(
                {
                    "exposure_count": 1,
                    "is_new_member": bool(new_of_slot[row][column]),
                }
            )
    new_share = new_member_exposure_share(slot_records)

    # gini over the per item exposure counts (how many cards exposed the item)
    flat_items = item_of_slot[valid_mask]
    counts = np.bincount(flat_items.astype(np.int64), minlength=batch["n_items"])
    gini = exposure_gini([{"exposure_count": int(value)} for value in counts])

    # NDCG per card
    ndcg_values = []
    for row in range(n_cards):
        row_rel = rel_of_slot[row][valid_mask[row]]
        ideal = sorted(
            (float(cand["rel"]) for cand in (dataset[row].get("candidates") or [])),
            reverse=True,
        )
        ndcg_values.append(ndcg_at_k(list(row_rel), ideal, k))
    ndcg_mean = float(np.mean(ndcg_values)) if ndcg_values else 0.0

    result = {
        "ndcg_mean": ndcg_mean,
        "coverage": float(coverage),
        "gini": float(gini),
        "new_share": float(new_share),
        "n_cards": int(n_cards),
        "n_slots": int(len(slot_records)),
    }
    if include_counts:
        result["exposure_counts"] = {
            batch["item_ids"][position]: int(counts[position])
            for position in range(batch["n_items"])
            if counts[position] > 0
        }
    return result


def current_metrics(
    dataset: List[Dict],
    current_weights: Sequence[float],
    k: int = DEFAULT_K,
    include_counts: bool = False,
) -> Dict[str, object]:
    """Metrics of the current weight vector (used to derive the thresholds)."""
    return evaluate_weights(
        dataset, current_weights, k=k, include_counts=include_counts
    )
