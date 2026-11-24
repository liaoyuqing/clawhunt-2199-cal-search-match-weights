# 验收证据索引

表中每个「断言位置」都是在当前 `HEAD` 上用 `git ls-tree -r HEAD --name-only` 核对过存在、
并用行号定位到具体断言行/函数定义行的真实引用；`python -m pytest -o addopts= -q`
总计 44 passed。

| 验收编号 | 复现命令 | 预期可观察结果 | 证据文件路径 | 断言所在位置 |
| --- | --- | --- | --- | --- |
| AC1 全部输出候选满足 4 项约束 | `python -m pytest -o addopts= -q tests/test_ac1_constraints.py` | 6 passed | `tests/test_ac1_constraints.py` | `tests/test_ac1_constraints.py:39` |
| AC2 同一种子重复运行得到相同的前 5 组候选 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py` | 6 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:33` |
| AC3 输出中包含被约束淘汰的最优解及其淘汰原因 | `python -m pytest -o addopts= -q tests/test_ac3_best_rejected.py` | 6 passed | `tests/test_ac3_best_rejected.py` | `tests/test_ac3_best_rejected.py:23` |
| 附加：权重网格计数（含与提案 2751 的差异复算） | `python -m pytest -o addopts= -q tests/test_search_space_and_metrics.py` | 13 passed | `tests/test_search_space_and_metrics.py` | `tests/test_search_space_and_metrics.py:10` |
| 附加：交叉验证与约束闸门 | `python -m pytest -o addopts= -q tests/test_cv_and_constraints.py` | 7 passed | `tests/test_cv_and_constraints.py` | `tests/test_cv_and_constraints.py:12` |
| 附加：坐标上升只接受合规点 | `python -m pytest -o addopts= -q tests/test_ascent_invariants.py` | 3 passed | `tests/test_ascent_invariants.py` | `tests/test_ascent_invariants.py:78` |
| 附加：密钥扫描在无 .git 时必须失败而非假绿 | `python -m pytest -o addopts= -q tests/test_secret_scan_guard.py` | 3 passed | `tests/test_secret_scan_guard.py` | `tests/test_secret_scan_guard.py:48` |
| 附加：全量用例 | `python -m pytest -o addopts= -q` | 44 passed；末行见 `reports/pytest_output.txt` | `reports/pytest_output.txt` | `reports/metrics.json` 的 `test_cases` / `collected_tests` 字段 |
| 附加：命令行输出与运行记录 | `python -m calib.run --seed 20261121 --restarts 20` | 写出 `calib_runs/run_seed20261121_r20.json` | `calib_runs/run_seed20261121_r20.json` | `calib/run.py:100` |
| 附加：同种子重复运行文件逐字节一致 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py::test_two_cli_runs_write_byte_identical_files` | 1 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:46` |
| 附加：运行不改线上权重快照 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py::test_cli_runs_do_not_touch_the_current_weights_snapshot` | 1 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:58` |
| 附加：同种子被评估点集完全相同 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py::test_same_seed_evaluates_exactly_the_same_points` | 1 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:79` |
| 附加：不同种子的被评估点集不同 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py::test_different_seeds_do_explore_different_points` | 1 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:92` |
| 附加：NDCG 曲面穷举（局部极大值个数） | `python scripts/ndcg_surface.py` | `local_maxima 1` | `reports/ndcg_surface.md` | `scripts/ndcg_surface.py:54` |
| 附加：实测数字汇总 | `python scripts/collect_metrics.py` | 打印 13 项指标 + 每个 run 一行 | `reports/metrics.json` | `scripts/collect_metrics.py:179` |
| 附加：密钥扫描 | `python scripts/secret_scan.py` | `tree_hits 0`、`history_hits 0`、`history_scanned yes` | `reports/metrics.json` | `scripts/secret_scan.py:96` |

## AC1 的具体断言

`tests/test_ac1_constraints.py:39` 的 `test_every_top5_candidate_passes_all_four_constraints`
对 `top5` 的每一条重新调用 `calib.constraints.evaluate`，要求 `passed is True`
且 `violations == []`；同文件另外 5 条断言 `top5` 非空、按 NDCG 降序、每个权重都落在
搜索网格内（`space.enumerate_points(4)` 的集合内）、权重之和为 1、单项不超过 0.5、
`best_rejected` 不出现在 `top5` 中，以及 `max_delta` 不越过 `item_delta_max`。

## AC2 的具体断言

`tests/test_ac2_reproducible.py` 里六件事：

1. `:33` `test_same_seed_gives_the_same_top5_weights`：同一 seed 两次搜索，`top5` 的
   权重与 NDCG 逐项相等；
2. `:40` `test_same_seed_gives_the_same_rejected_solution`：同一 seed 的
   `best_rejected` 整体字典相等；
3. `:46` `test_two_cli_runs_write_byte_identical_files`：两次调用命令行入口写出同一个
   JSON 文件，字节完全相同；
4. `:79` `test_same_seed_evaluates_exactly_the_same_points`：同一 seed 两次运行**被评估
   点集完全相同**（不止 top 5 相同）；
5. `:92` `test_different_seeds_do_explore_different_points`：不同 seed 的 `top5` 相同、
   但**被评估点集不同**；
6. `:58` `test_cli_runs_do_not_touch_the_current_weights_snapshot`：命令行运行前后
   `fixtures/current_weights.json` 的 mtime 与 sha256 不变，且运行记录里
   `online_weights_written` 为 `false`。

第 4、5 条是必要的：如果只断言 top 5 相同，一个恒定返回同一答案的搜索也能通过。
而「不同 seed 收敛到同一 top5」是这个夹具的数据性质而非搜索退化——
`reports/ndcg_surface.md` 穷举 891 个合法点，证明 NDCG 曲面只有 **1 个**局部极大值
（坐标 `[9, 4, 4, 3]`，NDCG 0.978106），因此任何起点都必然收敛到同一点；
能证明随机重启真实生效的正是被评估点集。实测：同一 seed 两次都是 430 个点且集合相等；
seed 7 是 482 个点，与 seed 20261121 的对称差为 268。

## AC3 的具体断言

`tests/test_ac3_best_rejected.py` 断言：`best_rejected` 非空（`:23`）、`rejected_reason` 与
`rejected_reason_text` 非空（`:28`）、原因顺序符合 `calib.constraints.CONSTRAINT_NAMES` 的
固定顺序（`:36`）、重新调用 `evaluate` 得到的违反集合与报告的 `rejected_reason` 完全一致
（`:44`）、`excess_total` 等于各条 excess 之和（`:55`），并且 `best_rejected` 的 NDCG 等于
全部被淘汰点里的最大值（`:62`）。
## 坐标上升只能返回合规点

`tests/test_ascent_invariants.py:78` 对全部 891 个合法起点各跑一次上升，断言返回点
`passed is True`，并断言网格里确实存在不可行起点（否则该断言等于空转）；
`:103` 用桩评估器把全局最优放在不可行的 `[0.5, 0.2, 0.15, 0.15]` 上，断言上升不会
把它当成 incumbent；`:136` 是正对照，桩的最优点合规时必须能走到，防止「永不动弹」
也能通过前一条。

为什么需要 `:103` 这条桩用例：在真实夹具上，即使把可行性判断整个删掉，891 个起点
也都会落在合规点，因为曲面唯一的局部极大值恰好合规。只看真实数据的 `:78` 无法
区分「校验了合规」与「碰巧合规」。反方向也一样：只加守卫、不加跨界，891 个起点里
有 569 个会卡在不可行起点上——守卫与跨界必须同时存在，因此三条用例缺一不可。

## 端到端自检

```bash
python scripts/make_fixtures.py       # 重建虚构夹具（幂等）
python -m pytest -o addopts= -q       # 44 passed
python -m calib.run --seed 20261121 --restarts 20
python scripts/collect_metrics.py
python scripts/ndcg_surface.py
python scripts/secret_scan.py         # history_scanned yes
```

在**不是** git 工作树的目录里运行 `scripts/secret_scan.py` 会打印 `history_scanned no`
并以退出码 2 结束，不会以 0 退出冒充通过（`tests/test_secret_scan_guard.py:48`）。
