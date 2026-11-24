"""Dataset loading helpers.

The fixtures live in `fixtures/` inside the repository. The directory can be
pointed somewhere else with the environment variable `CALIB_FIXTURES_DIR`
(used by the README runbook and by the tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES_DIR = REPO_ROOT / "fixtures"
FIXTURES_DIR_ENV = "CALIB_FIXTURES_DIR"

CARDS_FILENAME = "demand_cards_500.jsonl"
WEIGHTS_FILENAME = "current_weights.json"


def fixtures_dir() -> Path:
    """Return the fixture directory, honouring `CALIB_FIXTURES_DIR`."""
    override = os.environ.get(FIXTURES_DIR_ENV)
    if override:
        return Path(override)
    return DEFAULT_FIXTURES_DIR


def cards_path(path: Optional[str] = None) -> Path:
    """Resolve the demand card jsonl path."""
    if path is not None:
        return Path(path)
    return fixtures_dir() / CARDS_FILENAME


def weights_path(path: Optional[str] = None) -> Path:
    """Resolve the current weights snapshot path."""
    if path is not None:
        return Path(path)
    return fixtures_dir() / WEIGHTS_FILENAME


def load_dataset(path: Optional[str] = None) -> List[Dict]:
    """Load the demand cards.

    Each returned dict is the original json record plus one extra key
    `features_matrix` holding a numpy array of shape (n_candidates, 4).
    """
    target = cards_path(path)
    dataset: List[Dict] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            matrix = np.asarray(
                [cand["features"] for cand in card.get("candidates", [])], dtype=float
            )
            card["features_matrix"] = matrix
            dataset.append(card)
    return dataset


def load_current_weights(path: Optional[str] = None) -> Tuple[float, ...]:
    """Load the read-only current weight snapshot as a float tuple."""
    target = weights_path(path)
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = payload["weights"]
    return tuple(float(w) for w in weights)


def group_ids(dataset: List[Dict]) -> List[str]:
    """Sorted list of the distinct `group_id` values of the dataset."""
    return sorted({str(card["group_id"]) for card in dataset})


def strip_arrays(card: Dict) -> Dict:
    """Return a json-serialisable copy of a card (drops `features_matrix`)."""
    clean = {key: value for key, value in card.items() if key != "features_matrix"}
    return clean
