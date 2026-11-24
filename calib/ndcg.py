"""NDCG@10 with exponential gain.

DCG = sum over i (2 ** rel_i - 1) / log2(i + 2), with i starting at 0.
NDCG = DCG / IDCG, where IDCG uses the ideal (descending) relevance list.
"""

from __future__ import annotations

import math
from typing import Sequence


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """Discounted cumulative gain of the first k relevance values."""
    if k <= 0:
        return 0.0
    total = 0.0
    for i, rel in enumerate(list(relevances)[:k]):
        gain = (2.0 ** float(rel)) - 1.0
        total += gain / math.log2(i + 2.0)
    return total


def idcg_at_k(ideal_relevances: Sequence[float], k: int = 10) -> float:
    """Ideal DCG for the given relevance values (used as the denominator)."""
    return dcg_at_k(ideal_relevances, k)


def ndcg_at_k(
    ranked_relevances: Sequence[float],
    ideal_relevances: Sequence[float],
    k: int = 10,
) -> float:
    """NDCG@k of a ranked list against an ideal list.

    Returns 0.0 when the ideal list is empty or has zero gain.
    """
    ideal = list(ideal_relevances)
    if not ideal:
        return 0.0
    idcg = idcg_at_k(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / idcg
