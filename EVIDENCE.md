# 验收证据索引

| 验收编号 | 复现命令 | 预期可观察结果 | 证据文件路径 | 断言所在位置 |
| --- | --- | --- | --- | --- |
| AC1 全部输出候选满足 4 项约束 | `python -m pytest -o addopts= -q tests/test_ac1_constraints.py` | 6 passed | `tests/test_ac1_constraints.py` | `tests/test_ac1_constraints.py:39` |
| AC2 同一种子重复运行得到相同的前 5 组候选 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py` | 4 passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:33` |
| AC3 输出中包含被约束淘汰的最优解及其淘汰原因 | `python -m pytest -o addopts= -q tests/test_ac3_best_rejected.py` | 6 passed | `tests/test_ac3_best_rejected.py` | `tests/test_ac3_best_rejected.py:23` |
| 附加：权重网格计数 | `python -m pytest -o addopts= -q tests/test_search_space_and_metrics.py` | 13 passed | `tests/test_search_space_and_metrics.py` | `tests/test_search_space_and_metrics.py:10` |
| 附加：交叉验证与约束闸门 | `python -m pytest -o addopts= -q tests/test_cv_and_constraints.py` | 7 passed | `tests/test_cv_and_constraints.py` | `tests/test_cv_and_constraints.py:12` |
| 附加：全量用例 | `python -m pytest -o addopts= -q` | 36 passed | `reports/metrics.json` | `scripts/collect_metrics.py:138` |
| 附加：命令行输出与运行记录 | `python -m calib.run --seed 20261121 --restarts 20` | 写出 `calib_runs/run_seed20261121_r20.json` | `calib_runs/run_seed20261121_r20.json` | `calib/run.py:100` |
| 附加：同种子重复运行文件逐字节一致 | `python -m pytest -o addopts= -q tests/test_ac2_reproducible.py::test_two_cli_runs_write_byte_identical_files` | passed | `tests/test_ac2_reproducible.py` | `tests/test_ac2_reproducible.py:46` |
| 附加：实测数字汇总 | `python scripts/collect_metrics.py` | 打印 12 项指标 | `reports/metrics.json` | `scripts/collect_metrics.py:196` |
| 附加：密钥扫描 | `python scripts/secret_scan.py` | `tree_hits 0`、`history_hits 0` | `reports/metrics.json` | `scripts/secret_scan.py:66` |

## AC1 的具体断言

`tests/test_ac1_constraints.py::test_every_top5_candidate_passes_all_four_constraints`
对 `top5` 的每一条重新调用 `calib.constraints.evaluate`，要求 `passed is True`
且 `violations == []`；同时断言 `top5` 按 NDCG 降序、每个权重都落在搜索网格内
（`space.enumerate_points(4)` 的集合内）、权重之和为 1、单项不超过 0.5，
且 `best_rejected` 不出现在 `top5` 中。

## AC2 的具体断言

`tests/test_ac2_reproducible.py` 里三件事：

1. `test_same_seed_gives_the_same_top5_weights`：同一 seed 两次搜索，`top5` 的
   权重与 NDCG 逐项相等；
2. `test_same_seed_gives_the_same_rejected_solution`：同一 seed 的
   `best_rejected` 整体字典相等；
3. `test_two_cli_runs_write_byte_identical_files`：两次调用命令行入口写出同一个
   JSON 文件，字节完全相同。

## AC3 的具体断言

`tests/test_ac3_best_rejected.py` 断言：`best_rejected` 非空、`rejected_reason` 与
`rejected_reason_text` 非空、`violations` 非空、原因顺序符合
`calib.constraints.CONSTRAINT_NAMES` 的固定顺序、重新调用 `evaluate` 得到
的违反集合与报告的 `rejected_reason` 完全一致、`excess_total` 等于各条 excess 之和，
并且 `best_rejected` 的 NDCG 等于全部被淘汰点里的最大值。

## 端到端自检

```bash
python scripts/make_fixtures.py       # 重建虚构夹具
python -m pytest -o addopts= -q       # 36 passed
python -m calib.run --seed 20261121 --restarts 20
python scripts/collect_metrics.py
python scripts/secret_scan.py
```