# 内部开发协作规范（COMMON.md）

> ⚠️ **这是项目开发期的内部协作文档，不是给外部用户看的。**
> 其中包含本机专属的绝对路径（`E:\...`）与沙箱约束。
> 外部读者请看根目录 `README.md`；本机运行差异见 `docs/SANDBOX_NOTES.md`。

复现对象：Calvano, Calzolari, Denicolò & Pastorello (2020),
**"Artificial Intelligence, Algorithmic Pricing, and Collusion"**, *AER* 110(10): 3267–3297.

## 0. 运行环境（重要）

- 本机 **PyPI 不可访问**，DSH 自带 python 没有 matplotlib。
- 请统一使用系统 Python：
  `C:\Users\91203\AppData\Local\Programs\Python\Python312\python.exe`
  （含 numpy 1.26.4 / matplotlib 3.11 / scipy / networkx / pandas）
- 运行前设置 matplotlib 配置目录（否则沙箱会因 AppData 写权限报错）：
  ```powershell
  $env:MPLCONFIGDIR='E:\DeepSeek工作产出\人工智能与经济管理研究课程\replication\.mplconfig'
  ```
- 建议脚本用 `if __name__ == "__main__":` 保护（Windows 进程池用 spawn）。
- 中文控制台输出可能显示为乱码，属 PowerShell 5.1 编码问题，不影响结果。

## 1. 标定（已三重验证，见 tests/test_core.py）

论文正文写需求为 `exp(mu (a_i - p_i))` 且基准 `mu = 1/4`。**字面读法无法复现论文数字**：
按字面读法 Bertrand-Nash 利润率约 500%（p^N≈6.06）、垄断价≈7.19，与论文报告的
"47%" 和表 2 的价格网格完全不符。

采用 `exp((a_i - p_i)/mu)`（即指数系数 kappa = 1/mu = 4）后，三处独立吻合：

| 论文原文 | 本工程计算 |
|---|---|
| 静态 Bertrand 价-成利润率 "approximately 47 percent" | 0.4729 |
| 完全垄断时 "about twice as large" | 0.9250 |
| 表 2 偏离价 1.43/1.51/1.58/1.66/1.74/1.82/1.89 | 1.4277/1.5052/1.5827/1.6602/1.7377/1.8152/1.8927 |
| 表 2 冲击前价 1.62…1.97（步长 0.04） | 1.6215/1.6602/1.6990/1.7377/1.7764/1.8152/1.8539/1.8927/1.9314/1.9702 |
| "Δ≈−2% 若 Nash 由 defect 逼近；≈12% 若由 excess 逼近" | −2.38% / +11.66% |

`core.MarketSpec(exponent="div")` 为默认；`exponent="mul"` 保留字面读法。

## 2. 基准参数（第 II.E 节）

n=2, c_i=1, a_i−c_i=1（即 a_i=2）, a_0=0, mu=1/4, delta=0.95, m=15, xi=0.1, k=1；
代表性实验 alpha=0.15, beta=4e-6（每个格子因随机探索平均被访问约 nu≈20 次）。

离散收益表 `env["pi_table"]` 形状 (m^n, n)，行号 = 动作剖面的 base-m 编码
（`core.profile_index(actions, m)`，第 i 位对应第 i 家企业的动作，最右位变化最快）。

## 3. 公共 API

```python
from core import MarketSpec, benchmark_prices, price_grid, payoff_table, profile_index
from engine import (SimConfig, build_env, run_batch, run_chunk, merge,
                    CYCLE_CONST_SYM, CYCLE_CONST_ASYM, CYCLE_2, CYCLE_GE3, CYCLE_ORDER)
from parallel import run_config, run_configs, DEFAULT_WORKERS
```

`build_env(cfg) -> dict`: `grid (m,)`、`pi_table (m^n, n)`、`q0 (n, m)`、
`p_nash`、`p_mono`、`pi_nash`（双寡头每企业平均）、`pi_mono`、`benchmarks`。

`SimConfig` 关键字段：`spec, m, xi, grid_mode, delta, memory(k), alpha, beta,
patience, max_periods, fixed_periods, profit_window, exploration, init, demand_shock,
record_every, dtype, label`。

`run_batch(cfgs, sessions, seed, env) -> SessionResult`（同环境多点批处理）
`run_chunk(cfg, sessions, seed, env) -> SessionResult`
`run_config(cfg, n_sessions, n_workers)` / `run_configs(cfgs, n_sessions, n_workers)`：多进程。

`SessionResult` 字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `converged` | (N,) bool | 是否满足收敛判据 |
| `t_conv` | (N,) int | 收敛期数（未收敛 −1） |
| `pi_final` | (N,) | 收敛时每企业平均利润 |
| `delta()` | (N,) | 式 (9) 归一化超额利润 |
| `longrun_actions` | (N, n) | 稳态动作 |
| `longrun_prices` | (N, n) | 稳态价格 |
| `cycle_type` | (N,) | 0=1-Sym. 1=1-Asym. 2=周期2 3=周期≥3 4=未收敛 |
| `limit_actions` | (N, n, M) int16 | **极限策略** F_i(s)=argmax_a Q_i(s,a)，M=m^(n·k) |
| `onpath_states` | (N, W) int32 | 末端在轨状态序列（环形缓冲，长度 W=profit_window） |
| `env`, `meta` | dict | 环境与元信息 |
| `history` | dict 或 None | `record_every>0` 时有 `pi (N,T)`、`eps (N,T)` |

## 4. 极限策略的用法（第 IV 节的核心）

`s = profile_index([a_1, a_2], m)` 即状态；双方按极限策略推进时下一状态为
`next_state = profile_index(limit_actions[r, :, s], m)`。给定对手的极限策略，
本方的"真 Q 值"由线性方程 `Q(s,a) = pi(s,a) + delta * max_{a'} Q(s',a')` 决定
（`s' = profile_index([a, a_j^L(s)], m)`），可用值迭代求解。

## 5. 规模与耗时（务必遵守）

- 单个会话需要 **200 万～400 万期** 才收敛（β=4e-6 时 ε 到 1e-3 需 ~69 万期）。
- numpy 内层循环每周期固定开销约 100–200 μs，**批量至少 200 行**才划算。
- 参考吞吐：约 4×10^5 行·期 / 秒 / 进程，20 进程并行。
- 结果写入 `replication/results/`，图写入 `replication/figures/`。
- 禁止修改 `src/core.py`、`src/engine.py`、`src/parallel.py`（公共基础设施）。
  如确需改动，先在结果文件里说明。

## 6. 论文中的目标数值（用于对照）

- Δ：网格 100×100 上 0.70–0.90；代表性实验（α=0.15, β=4e-6）Δ≈0.85；
  纯 Nash 均衡会话 Δ≈0.854；`delta` 降到 0.35 时 Δ≈0.16。
- 表 1：1-Sym 0.277、1-Asym 0.366、周期2 0.238、周期≥3 0.119；平均 Δ 0.849；
  on-path Nash 频率 0.505；on-path Q-loss 0.002、全状态 Q-loss 0.018。
- 图 2/图 7：β 越大（探索越少）Nash 频率越低、惩罚越弱。
- 惩罚强度与利润增益的相关系数 0.762；Δ 与 Nash 频率相关系数 0.12。
- 第 V 节：n=3 Δ≈0.64、n=4 Δ≈0.56（降低 β 后 n=3 可回升到 ≈0.75）；
  成本不对称（c_1=1）Δ 从 0.849 降到 0.713；随机需求 a_0∈{−0.15,0,0.15} → Δ≈0.80，
  a_0∈{−0.25,0,0.25} → Δ≈0.70；进入退出 → Δ<0.60；μ=0（完全替代）Δ 仍 >77%；
  m=100 → Δ≈0.70。
- 表 4（c_1=1, c_2=1/0.875/0.75/0.625/0.5/0.25）：
  Nash 份额 0.500/0.545/0.588/0.627/0.662/0.722；Δ 0.849/0.841/0.812/0.781/0.759/0.713；
  π_1/π_1^N 1.000/1.050/1.121/1.193/1.265/1.442。
- 图 8/9：92% 的会话从任意节点都能回到长期价格节点；98% 的会话中不收敛节点少于 3 个；
  平均路径长度 6，最大 18。
- 图 10：利润增益远早于收敛就上升（相对"外生探索"基准线）；图 11：重新配对后 Δ 从 0.85
  掉到约 0.20，随后在不到原来 1/10 的时间内回升。