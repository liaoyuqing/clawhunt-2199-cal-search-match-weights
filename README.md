# 匹配权重约束搜索与曝光公平约束（cal search-match-weights）

本仓库实现会员空间平台「搜索匹配权重」的**受约束校准搜索**：在固定的权重网格上
用坐标上升 + 随机重启寻找 NDCG@10 最优的候选权重，并且只保留同时满足四项公平
约束的候选；同时输出**被约束淘汰的最优解**及其淘汰原因。

> 全部数据均为虚构夹具（`fixtures/`），不含任何真实客户名称、商品名称或个人信息；
> 本仓库不会写回线上权重，也不向任何外部服务发送数据。

## 1. 一句话口径与两个必要假设

工单原文没有写死「匹配特征分项数」，本实现按 **k = 4 个匹配特征分项**落地
（`calib/space.py` 的 `k=4` 计数为 891 个合法点）。README 顶部与
`reports/metrics.json` 的 `search_space` 字段都写明了这个假设，读代码的人
不需要反推。

第二个假设是**曝光口径**：工单里的「曝光基尼」按「某个商品出现在多少张
需求卡的 top10 里」统计。因此夹具里的商品必须**跨需求卡复用**（共享商品目录），
否则每个商品只会被曝光 0 或 1 次，基尼系数恒等于 0.5，约束会退化成永远成立
或永远失败的死约束。夹具用 10000 个 id 的共享目录池，实际出现 4472 个不同商品。

## 2. 运行环境

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -o addopts= -q
.venv/Scripts/python -m calib.run --seed 20261121 --restarts 20
```

实测环境：Python 3.11.9、numpy 2.4.6、pytest 9.2.0，Windows 与 Linux 均可运行；
不访问网络、不依赖数据库、不需要密钥。

## 3. 搜索空间

| 项 | 值 |
| --- | --- |
| 分项数 k | 4 |
| 单项范围 | 0 ~ 0.5 |
| 步长 | 0.05 |
| 合计 | 恰为 1.0（等式约束） |
| 合法点数（合计恰为 1） | 891 |
| 合法点数（合计 ≤ 1） | 7766 |

内部用**整数步长单位**表示权重（1 单位 = 0.05，每项 0~10 单位，合计 20 单位），
这样「合计为 1」是整数等式，不会有浮点漂移。计数用滚动数组 DP 求得，`k=3/4/5`
三个参考值与暴力枚举一致（66 / 891 / 7051）。

## 4. 目标函数与四项约束

目标：最大化 **NDCG@10**（`DCG = Σ (2^rel − 1) / log2(i + 2)`，`i` 从 0 开始，
除以该需求卡理想排序的 IDCG）。

四项约束（任一不满足即淘汰该候选点）：

| 约束 | 判定式 |
| --- | --- |
| 供给覆盖率 | `coverage >= 当前 coverage − 0.05` |
| 曝光基尼 | `gini <= 当前 gini + 0.05` |
| 新会员曝光占比 | `new_share >= 当前 new_share` |
| 分项相对变化 | `max|w_i − w_i_current| <= 0.2` |

淘汰原因的顺序是**固定**的：`coverage → gini → new_share → item_delta`。

## 5. 搜索方法

因为合计是等式约束，一次「坐标移动」等价于**把一个单位从一个分项转移到另一个
分项**（步长 0.05）。算法为坐标上升 + 20 次随机重启，固定种子：

1. `calib/space.py::enumerate_points` 给出排序后的全部合法点（k=4 时 891 个）；
2. `calib/search.py::random_restarts` 用 `np.random.default_rng(seed)` 在这些点里
   抽 20 个互不相同的起点（外加当前权重所在点作为一次起点）；
3. `calib/search.py::coordinate_ascent` 从每个起点出发，每轮枚举「单位转移」
   邻居，取 NDCG 最高的点前进，无改进即停（最多 50 轮）；
4. 搜索过程中**每一个被约束淘汰的点都被记录**，最终 NDCG 最高的那个成为
   `best_rejected`；
5. `top5` **只收合规点**，按 NDCG 降序。

`best_rejected` 与 `top5` 不会重叠（`search_weights` 里有断言），
`best_rejected` 带 `rejected_reason`（有序约束名）、`rejected_reason_text`（中文可读
描述）、`violations`（每条含 constraint / value / limit / excess）与 `excess_total`。

## 6. 5 折交叉验证

按 `group_id` 分组：`g-01 … g-10` 共 10 组，两两配对成 5 折，每折 2 组 /
100 张需求卡，折间分组两两不相交。`calib/cv.py::cross_validate` 对每折报告
NDCG@10 与三项指标，汇总给出均值与**总体标准差**（`np.std(ddof=0)`）。

## 7. 输出

`calib/run.py` 写出 `calib_runs/{run_id}.json`，`run_id = run_seed{seed}_r{restarts}`，
**只由 seed 与 restarts 决定**，所以同参数重复运行的输出文件逐字节一致。

JSON 关键字段：`top5`（weights / ndcg / coverage / gini / new_share / max_delta /
diff_vs_current）、`best_rejected`、`cv`（5 折 + 均值标准差）、`thresholds`、
`current_metrics`、`evaluated_points`、`search_space`、`online_weights_written`
（**恒为 false**）。

`generated_at` 是固定字符串而非时间戳，这是为了让「同种子重复运行得到相同结果」
这条验收可以在文件层面直接断言。

## 8. 只写校准记录，不动线上

- `fixtures/current_weights.json` 是**只读快照**，代码里没有任何写回路径；
- `calib.run` 只写 `calib_runs/`；
- `online_weights_written` 字段恒为 `false`，测试 `test_cli_runs_do_not_touch_the_current_weights_snapshot`
  断言两次运行前后该文件字节不变；
- 不包含任何向外部发送数据的第三方 SDK，运行时只读本地夹具。

## 9. 对接点（与其他工单的接口）

本仓库是纯计算模块，不直接连数据库。真实接入时按下列桩位替换即可：

| 位置 | 现在 | 生产环境替换为 |
| --- | --- | --- |
| `calib/data.py::load_dataset` | 读 `fixtures/demand_cards_500.jsonl` | 从搜索服务/特征库拉取需求卡与候选特征 |
| `calib/data.py::load_current_weights` | 读 `fixtures/current_weights.json` | 从配置中心读线上生效权重 |
| `calib/scoring.py::evaluate_weights` | 本地 numpy 计算 | 可换成向量化的离线特征服务（输入输出契约不变） |

卡片字段契约：`demand_card_id`、`group_id`、`candidates[].{item_id, features[4], rel,
has_supply, is_new_member}`。

## 10. 目录结构

```
calib/
  space.py        权重网格（整数步长单位）、计数、权重换算
  ndcg.py         NDCG@10 / DCG / IDCG
  metrics.py      供给覆盖率、曝光基尼、新会员占比、分项变化
  constraints.py  四项约束闸门、阈值推导、固定顺序的淘汰原因
  data.py         夹具加载（环境变量 CALIB_FIXTURES_DIR 可覆盖）
  scoring.py      单卡排序、NDCG、整库指标评估
  search.py       坐标上升、随机重启、top5 与 best_rejected
  cv.py           分组 5 折交叉验证
  run.py          命令行入口
scripts/
  make_fixtures.py    生成虚构夹具（幂等）
  collect_metrics.py  采集实测数字到 reports/metrics.json
  secret_scan.py      工作树与 git 历史的密钥扫描
tests/                验收断言与单元测试
fixtures/             虚构夹具（只读）
calib_runs/           校准运行记录（只写这里）
```

## 11. 已知边界

- 搜索是局部启发式的（坐标上升），不保证全局最优；本仓库报告的是「搜索到的最优」；
- 基尼系数用「曝光次数」计算，若商品目录不跨卡复用，该指标会退化（见第 1 节）；
- 夹具是虚构数据，真实分布下四项阈值需要按线上当前指标重新推导；
- `rel` 标签由夹具生成器的四分位切分产生，生产环境应使用真实反馈标签。
