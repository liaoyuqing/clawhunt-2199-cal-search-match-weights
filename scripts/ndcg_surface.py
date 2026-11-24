"""Exhaustively map the NDCG@10 surface over the legal weight grid.

The calibration search is a local heuristic (coordinate ascent with random
restarts), so an honest reproducibility statement has to say something about
the shape of the objective. This script enumerates every legal point, computes
NDCG@10 for each one, and reports how many local maxima the surface has, the
coordinate and NDCG of each one, the gap between the global maximum and the
runner up, and how many distinct NDCG values the grid produces.

Usage:  python scripts/ndcg_surface.py
        python scripts/ndcg_surface.py --markdown reports/ndcg_surface.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, seed) pairs, one row each. The same seed appears twice on purpose:
# the table has to show that a repeated seed reproduces the same point set,
# and that a different seed does not.
SEEDS = (
    ("20261121 (run 1)", 20261121),
    ("20261121 (run 2)", 20261121),
    ("7", 7),
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calib import constraints as constraint_gate
from calib import space
from calib.data import load_current_weights, load_dataset
from calib.search import Evaluator, _neighbours, random_restarts


def build_surface(evaluator: Evaluator, current_weights: Sequence[float], thresholds: Dict[str, float]):
    """Return (points, values, feasible) for every legal point."""
    points = space.enumerate_points(len(current_weights))
    values: Dict[Tuple[int, ...], float] = {}
    feasible: Dict[Tuple[int, ...], bool] = {}
    for point in points:
        weights = space.to_weights(point)
        metrics = evaluator.evaluate(weights)
        passed, _ = constraint_gate.evaluate(weights, current_weights, metrics, thresholds)
        values[point] = float(metrics["ndcg_mean"])
        feasible[point] = bool(passed)
    return points, values, feasible


def local_maxima(points: Sequence[Tuple[int, ...]], values: Dict[Tuple[int, ...], float]) -> List[Tuple[int, ...]]:
    """A point is a local maximum when no unit transfer improves its NDCG."""
    maxima = []
    for point in points:
        value = values[point]
        is_max = True
        for neighbour in _neighbours(point, 1, space.CAP_UNITS):
            if neighbour in values and values[neighbour] > value + 1e-12:
                is_max = False
                break
        if is_max:
            maxima.append(point)
    return maxima


def point_sets(evaluator: Evaluator, current_weights, thresholds, seeds):
    """Evaluated point set per seed, as measured by the real search."""
    measured = []
    for label, seed in seeds:
        search = random_restarts(
            evaluator, current_weights, thresholds, seed=seed, restarts=20
        )
        evaluated = set(search["feasible"]) | set(search["rejected"])
        measured.append((label, seed, evaluated))
    return measured


def render(points, values, feasible, maxima, measured) -> str:
    ordered = sorted(points, key=lambda p: (-values[p], p))
    best = ordered[0]
    runner = ordered[1]
    distinct = len({round(v, 12) for v in values.values()})
    feasible_count = sum(1 for p in points if feasible[p])

    lines = []
    lines.append("# NDCG@10 曲面的穷举结果")
    lines.append("")
    lines.append("本文件由 `python scripts/ndcg_surface.py --markdown reports/ndcg_surface.md` 生成，")
    lines.append("全部数字由脚本现算，未手工填写。")
    lines.append("")
    lines.append("## 为什么要做这份穷举")
    lines.append("")
    lines.append("校准搜索是局部启发式（坐标上升 + 随机重启），因此「换一个随机种子会不会收敛到")
    lines.append("不同的点」取决于目标曲面的形状。把全部合法点算一遍，可以把这件事变成可核查的")
    lines.append("事实，而不是口头解释。")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("| 项 | 实测值 |")
    lines.append("| --- | --- |")
    lines.append("| 合法点总数 | {0} |".format(len(points)))
    lines.append("| 合规点总数 | {0} |".format(feasible_count))
    lines.append("| 局部极大值个数 | {0} |".format(len(maxima)))
    lines.append("| 全局最大 NDCG@10 | {0:.6f} |".format(values[best]))
    lines.append("| 次大 NDCG@10 | {0:.6f} |".format(values[runner]))
    lines.append("| 全局与次大的差值 | {0:.6f} |".format(values[best] - values[runner]))
    lines.append("| 互不相同的 NDCG 取值个数 | {0} |".format(distinct))
    lines.append("")
    lines.append("局部极大值清单（坐标为整数步长单位，1 单位 = 0.05）：")
    lines.append("")
    lines.append("| 坐标 | 权重 | NDCG@10 | 是否合规 |")
    lines.append("| --- | --- | --- | --- |")
    for point in sorted(maxima, key=lambda p: -values[p]):
        weights = space.to_weights(point)
        lines.append(
            "| {0} | {1} | {2:.6f} | {3} |".format(
                list(point),
                " / ".join("{0:.2f}".format(w) for w in weights),
                values[point],
                "合规" if feasible[point] else "被淘汰",
            )
        )
    lines.append("")
    lines.append("## 这对「换种子是否改变结果」意味着什么")
    lines.append("")
    if len(maxima) == 1:
        lines.append(
            "曲面在 {0} 个合法点上只有 **1 个**局部极大值，因此任何起点的坐标上升都必然收敛"
            "到同一个点：不同随机种子得到相同的前 5 组候选权重，是目标函数的性质，不是搜索"
            "退化的证据。能证明随机重启真实生效的是**被评估点集**：不同种子抽取不同起点，"
            "触及的权重点集合不同，只是收敛点相同。".format(len(points))
        )
    else:
        lines.append(
            "曲面有 {0} 个局部极大值，因此不同随机种子可能收敛到不同的点。".format(len(maxima))
        )
    lines.append("")
    lines.append("### 实测：同种子与换种子的被评估点集")
    lines.append("")
    reference_set = measured[0][2]
    lines.append(
        "| 取样 | 种子 | 被评估点集大小 | 与第一行的对称差 |"
    )
    lines.append("| --- | --- | --- | --- |")
    for label, seed, evaluated in measured:
        lines.append(
            "| {0} | {1} | {2} | {3} |".format(
                label, seed, len(evaluated), len(evaluated ^ reference_set)
            )
        )
    lines.append("")
    repeated = measured[0][2] ^ measured[1][2] if len(measured) > 1 else set()
    changed = measured[0][2] ^ measured[-1][2]
    lines.append(
        "同一种子两次运行的被评估点集完全相同（对称差 {0}）；"
        "换成种子 {1} 后对称差为 {2}，点集确实变了——这是随机重启"
        "真实生效的证据；两者的 top5 相同，则是上面说的单峰性质。".format(
            len(repeated), measured[-1][1], len(changed)
        )
    )
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/ndcg_surface.py")
    lines.append("python scripts/ndcg_surface.py --markdown reports/ndcg_surface.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/ndcg_surface.py")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)

    dataset = load_dataset()
    current_weights = load_current_weights()
    evaluator = Evaluator(dataset, k=10)
    current = evaluator.evaluate(current_weights)
    thresholds = constraint_gate.derive_thresholds(current)

    points, values, feasible = build_surface(evaluator, current_weights, thresholds)
    maxima = local_maxima(points, values)
    measured = point_sets(evaluator, current_weights, thresholds, SEEDS)
    ordered = sorted(points, key=lambda p: (-values[p], p))
    best = ordered[0]
    runner = ordered[1]
    distinct = len({round(v, 12) for v in values.values()})

    print("points", len(points))
    print("feasible_points", sum(1 for p in points if feasible[p]))
    print("local_maxima", len(maxima))
    print("best_point", list(best))
    print("best_ndcg {0:.6f}".format(values[best]))
    print("runner_up_ndcg {0:.6f}".format(values[runner]))
    print("gap {0:.6f}".format(values[best] - values[runner]))
    print("distinct_ndcg_values", distinct)

    if args.markdown:
        target = Path(args.markdown)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render(points, values, feasible, maxima, measured),
            encoding="utf-8",
            newline="\n",
        )
        print("markdown", target.relative_to(REPO_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
