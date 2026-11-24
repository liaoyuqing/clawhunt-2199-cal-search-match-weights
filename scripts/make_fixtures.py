"""Generate the fictional fixtures for the calibration task.

Two artifacts are produced, both fully synthetic (no real company, person,
contact or production data appears anywhere):

  fixtures/demand_cards_500.jsonl  500 demand cards, 20 candidates each
  fixtures/current_weights.json     read-only snapshot of the current weights

Item model
----------
Items live in a shared catalogue of 10000 ids (it-000001 .. it-010000). Every
item has one fixed feature vector; a demand card references 20 of those items
as its candidates, so an item can compete in several demand cards. This
matters for the exposure gini: the gini is computed over "in how many cards
did this item reach the top 10", which is only sensitive to the weight vector
when items recur across cards.

Re-running the script is idempotent: the RNG seed is fixed, so the byte
content of both files is stable.

Usage:  python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261124
N_CARDS = 500
N_GROUPS = 10
CARDS_PER_GROUP = 50
CANDIDATES_PER_CARD = 20
N_FEATURES = 4
ITEM_POOL_SIZE = 10000
POPULARITY_EXPONENT = 0.75

# Direction of the matching signal that is shared by every demand card. A
# weight vector close to it ranks the candidates closer to the relevance
# labels, which is exactly what the calibration search is supposed to find.
GLOBAL_DIRECTION = np.array([0.55, 0.20, 0.15, 0.10], dtype=float)
# Share of the global direction inside the per card latent direction.
LATENT_BLEND = 0.7

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CARDS_PATH = FIXTURES_DIR / "demand_cards_500.jsonl"
WEIGHTS_PATH = FIXTURES_DIR / "current_weights.json"

CURRENT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


def build_item_catalogue(rng: np.random.Generator):
    """Return (item_ids, item_features, popularity) for the shared catalogue."""
    item_ids = ["it-%06d" % (index + 1) for index in range(ITEM_POOL_SIZE)]
    features = np.round(rng.random((ITEM_POOL_SIZE, N_FEATURES)), 2)

    # Long tail catalogue: a few head items compete in many cards, most items
    # appear in a few cards only. The exposure distribution therefore depends
    # on the ranking, which is what makes the gini constraint meaningful.
    ranks = np.arange(1, ITEM_POOL_SIZE + 1, dtype=float)
    popularity = ranks ** (-POPULARITY_EXPONENT)
    popularity = popularity / popularity.sum()

    return item_ids, features, popularity


def build_cards(rng: np.random.Generator) -> list:
    """Build the 500 fictional demand cards as plain dicts."""
    item_ids, item_features, popularity = build_item_catalogue(rng)
    cards = []

    for index in range(N_CARDS):
        card_no = index + 1
        demand_card_id = "dc-%04d" % card_no
        group_no = (card_no - 1) // CARDS_PER_GROUP + 1
        group_id = "g-%02d" % group_no

        chosen = rng.choice(
            ITEM_POOL_SIZE, size=CANDIDATES_PER_CARD, replace=False, p=popularity
        )
        chosen = np.sort(chosen)

        # Supply archetype: 40% of the cards are fully supplied, 40% are nearly
        # empty (0..4 supplied candidates), 20% are well supplied. The mix keeps
        # the overall supply rate near 60% while the card level coverage still
        # reacts to the candidate ranking.
        archetype = rng.random()
        if archetype < 0.40:
            n_supply = CANDIDATES_PER_CARD
        elif archetype < 0.80:
            n_supply = int(rng.integers(0, 5))
        else:
            n_supply = int(rng.integers(10, CANDIDATES_PER_CARD + 1))
        has_supply = np.zeros(CANDIDATES_PER_CARD, dtype=bool)
        if n_supply:
            has_supply[rng.choice(CANDIDATES_PER_CARD, size=n_supply, replace=False)] = True

        n_new = int(rng.binomial(CANDIDATES_PER_CARD, 0.25))
        is_new_member = np.zeros(CANDIDATES_PER_CARD, dtype=bool)
        if n_new:
            is_new_member[rng.choice(CANDIDATES_PER_CARD, size=n_new, replace=False)] = True

        card_features = item_features[chosen]

        # Relevance labels: per card latent weights over the 4 matching
        # features, quartiles of the resulting score (3 / 2 / 1 / 0).
        #
        # The latent direction is a blend of a global direction (shared by every
        # card, so a fixed weight vector can learn it) and a per card random
        # direction (so no single weight vector wins everywhere). With the
        # global part dominant the calibration has a real optimum to find.
        latent = LATENT_BLEND * GLOBAL_DIRECTION + (1.0 - LATENT_BLEND) * rng.dirichlet(
            np.ones(N_FEATURES)
        )
        scores = card_features @ latent
        order = np.lexsort((np.arange(CANDIDATES_PER_CARD), -np.round(scores, 12)))
        rel = np.zeros(CANDIDATES_PER_CARD, dtype=int)
        for rank, position in enumerate(order):
            rel[position] = 3 - rank // 5

        candidates = []
        for position in range(CANDIDATES_PER_CARD):
            item_position = int(chosen[position])
            candidates.append(
                {
                    "item_id": item_ids[item_position],
                    "features": [float(v) for v in card_features[position]],
                    "rel": int(rel[position]),
                    "has_supply": bool(has_supply[position]),
                    "is_new_member": bool(is_new_member[position]),
                }
            )

        cards.append(
            {
                "demand_card_id": demand_card_id,
                "group_id": group_id,
                "candidates": candidates,
            }
        )

    return cards


def write_cards(cards: list) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    with CARDS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False))
            handle.write("\n")


def write_current_weights() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": list(CURRENT_WEIGHTS),
        "recorded_at": "2026-11-20T00:00:00+08:00",
        "note": "fictional snapshot for calibration, read-only",
    }
    with WEIGHTS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    rng = np.random.default_rng(SEED)
    cards = build_cards(rng)
    write_cards(cards)
    write_current_weights()

    n_candidates = sum(len(card["candidates"]) for card in cards)
    groups = sorted({card["group_id"] for card in cards})
    supply_true = sum(
        1 for card in cards for c in card["candidates"] if c["has_supply"]
    )
    new_true = sum(
        1 for card in cards for c in card["candidates"] if c["is_new_member"]
    )
    distinct_items = len(
        {c["item_id"] for card in cards for c in card["candidates"]}
    )

    print("demand cards         : %d" % len(cards))
    print("candidates           : %d" % n_candidates)
    print("groups               : %d (%s .. %s)" % (len(groups), groups[0], groups[-1]))
    print("distinct items       : %d" % distinct_items)
    print("has_supply == true   : %d" % supply_true)
    print("is_new_member == true: %d" % new_true)
    print("cards written to     : %s" % CARDS_PATH)
    print("weights written to   : %s" % WEIGHTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
