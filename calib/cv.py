"""Grouped cross validation for the calibration run."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import group_ids
from .search import Evaluator


def group_folds(dataset: List[Dict], n_splits: int = 5) -> List[List[str]]:
    """Split the demand cards into disjoint groups per fold."""
    if int(n_splits) <= 0:
        raise ValueError("n_splits must be positive")
    groups = group_ids(dataset)
    folds: List[List[str]] = []
    size = len(groups)
    if size % int(n_splits) == 0:
        chunk = size // int(n_splits)
        for index in range(int(n_splits)):
            folds.append(groups[index * chunk:(index + 1) * chunk])
    else:
        for index in range(int(n_splits)):
            folds.append([groups[i] for i in range(index, size, int(n_splits))])
    return folds


def cross_validate(
    dataset: List[Dict],
    weights: Sequence[float],
    n_splits: int = 5,
    k: int = 10,
) -> Dict[str, object]:
    """5 fold cross validation grouped by demand card group."""
    folds = group_folds(dataset, n_splits)
    all_groups = set(group_ids(dataset))
    rows: List[Dict[str, object]] = []
    for index, test_groups in enumerate(folds):
        test_set = set(test_groups)
        test_cards = [c for c in dataset if str(c["group_id"]) in test_set]
        train_cards = [c for c in dataset if str(c["group_id"]) not in test_set]
        evaluator = Evaluator(test_cards, k=k)
        metrics = evaluator.evaluate(weights)
        train_evaluator = Evaluator(train_cards, k=k)
        train_metrics = train_evaluator.evaluate(weights)
        rows.append(
            {
                "fold": index + 1,
                "test_groups": list(test_groups),
                "train_groups": sorted(all_groups - test_set),
                "n_cards": len(test_cards),
                "n_train_cards": len(train_cards),
                "ndcg": float(metrics["ndcg_mean"]),
                "coverage": float(metrics["coverage"]),
                "gini": float(metrics["gini"]),
                "new_share": float(metrics["new_share"]),
                "train_ndcg": float(train_metrics["ndcg_mean"]),
            }
        )

    def mean_std(key: str):
        values = [float(row[key]) for row in rows]
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values, ddof=0))

    ndcg_mean, ndcg_std = mean_std("ndcg")
    coverage_mean, coverage_std = mean_std("coverage")
    gini_mean, gini_std = mean_std("gini")
    new_share_mean, new_share_std = mean_std("new_share")
    return {
        "n_splits": int(n_splits),
        "folds": rows,
        "ndcg_mean": ndcg_mean,
        "ndcg_std": ndcg_std,
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        "gini_mean": gini_mean,
        "gini_std": gini_std,
        "new_share_mean": new_share_mean,
        "new_share_std": new_share_std,
    }
