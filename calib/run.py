"""Command line entry point for the search-match-weight calibration run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import constraints as constraint_gate
from . import space
from .cv import cross_validate
from .data import load_current_weights, load_dataset
from .search import Evaluator, search_weights

DEFAULT_SEED = 20261121
DEFAULT_RESTARTS = 20
DEFAULT_K = 10
GENERATED_AT = "2026-11-24T00:00:00+08:00"
GENERATED_AT_NOTE = "fixed string so that two runs with the same seed give byte identical files"
REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calib.run",
        description="Constrained search for the search-match weights.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--fixtures-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixtures_dir:
        os.environ["CALIB_FIXTURES_DIR"] = args.fixtures_dir
    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=args.k)
    current = evaluator.evaluate(current_weights)
    thresholds = constraint_gate.derive_thresholds(current)
    result = search_weights(
        evaluator,
        current_weights,
        thresholds,
        seed=args.seed,
        restarts=args.restarts,
    )
    top5 = result["top5"]
    if top5:
        cv_weights = tuple(top5[0]["weights"])
        cv_source = "top1"
    else:
        cv_weights = tuple(float(w) for w in current_weights)
        cv_source = "current_weights"
    cv = cross_validate(dataset, cv_weights, n_splits=5, k=args.k)
    run_id = "run_seed{0}_r{1}".format(args.seed, args.restarts)
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "calib_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (run_id + ".json")
    payload = {
        "run_id": run_id,
        "seed": int(args.seed),
        "restarts": int(args.restarts),
        "k": int(args.k),
        "weights_current": [float(w) for w in current_weights],
        "thresholds": thresholds,
        "current_metrics": current,
        "top5": top5,
        "best_rejected": result["best_rejected"],
        "cv": cv,
        "cv_weights_source": cv_source,
        "evaluated_points": int(result["evaluated_points"]),
        "feasible_points": int(result["feasible_points"]),
        "rejected_points": int(result["rejected_points"]),
        "search_space": {
            "k": len(current_weights),
            "step": space.STEP,
            "cap_units": space.CAP_UNITS,
            "total_units": space.TOTAL_UNITS,
            "exact_points": space.count_points_exact(len(current_weights)),
            "le_points": space.count_points_le(len(current_weights)),
        },
        "online_weights_written": False,
        "generated_at": GENERATED_AT,
        "generated_at_note": GENERATED_AT_NOTE,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print("seed", args.seed)
    print("restarts", args.restarts)
    print("evaluated_points", result["evaluated_points"])
    print("feasible_points", result["feasible_points"])
    print("rejected_points", result["rejected_points"])
    print("top5")
    for index, entry in enumerate(top5):
        print(
            "  {0} weights={1} ndcg={2:.6f} coverage={3:.6f} gini={4:.6f} new_share={5:.6f} max_delta={6:.6f}".format(
                index + 1,
                entry["weights"],
                float(entry["ndcg"]),
                float(entry["coverage"]),
                float(entry["gini"]),
                float(entry["new_share"]),
                float(entry["max_delta"]),
            )
        )
    rejected = result["best_rejected"]
    if rejected is None:
        print("best_rejected none")
    else:
        print("best_rejected", rejected["weights"], "ndcg", round(float(rejected["ndcg"]), 6))
        print("best_rejected_reason", ",".join(rejected["rejected_reason"]))
        print("best_rejected_excess_total", round(float(rejected["excess_total"]), 6))
    print("cv_ndcg_mean", round(float(cv["ndcg_mean"]), 6))
    print("cv_ndcg_std", round(float(cv["ndcg_std"]), 6))
    print("output", _relative(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
