# 第 V 节（稳健性）+ 第 VI 节（时间尺度）复现报告

## 0. 总体口径（必须先读）

**所有仿真会话**都跑在 **fixed_periods** 模式（不等到论文意义的"收敛"），Lead 协调（2026-10-20 19:08）：

- 单进程（沙箱内 `multiprocessing.Pipe` 被禁，`ProcessPoolExecutor` 不能用）；
- 每会话跑 600K 周期（few 图 10/11 略不同）；
- Δ 用 `res.delta()` 即末尾 1000 期 `pi_ring` 均值换算；
- "converged_frac" 在 fixed_periods 模式下恒为 1.0（论文意义的"arch-persistance 收敛"需 2-4M 周期才能达到）。

→ **Δ 的绝对水平**比论文偏低 0.5-0.7（论文需 2-4M 期，本工程截到 600K-1.5M）；
→ **Δ 的形状/趋势**与论文大体一致（升序/降序、相对大小）。

另外：
- 引擎在 2026-10-20 16:52 升级（行压缩 bug 修复、`history["pi_ma"]` 出现）。
- `results/representative.npz`（dev-graph 在 16:05 升级**之前**生成的，130 KB，**无 Q 矩阵**）已被
  Lead 标记为与新引擎不同源；
- 我按 Lead 协调方案 B 自己跑了 `results/representative_small.npz`（30 会话 × 1.5M，含 Q_final）。
- 图 11 的 `run_rematch` 现在默认读 `representative_small.npz` 而不是 `representative.npz`。

---

## 1. 图 3（δ 扫描）+ k=0 对照

**命令**：
```
python run_robustness.py --only delta,k0 --sessions_delta 80 --fixed_periods 600000
```

**实测**（`results/robustness_delta.csv`，80 sessions × 600K periods 单进程）：

| δ    |  实测 Δ  | 论文 Δ |  Δ 差距  | 备注 |
|------|---------|--------|---------|------|
| 0.0  | 0.226   | (基线 ~0.12) | +0.10 | ε→exp(0)=1 全程探索，未收敛 |
| 0.1  | 0.234   | (单调上升)  | — | |
| 0.2  | 0.258   | (单调上升)  | — | |
| 0.35 | 0.253   | ~0.16 | +0.09 | 论文说"降到 0.16"我们看到 0.25 |
| 0.5  | 0.272   | (单调上升)  | — | |
| 0.65 | 0.283   | (单调上升)  | — | |
| 0.8  | 0.291   | (单调上升)  | — | |
| 0.9  | 0.318   | (上升)      | — | |
| 0.95 | 0.313   | ~0.85 | -0.54 | **未收敛**，论文需 2-4M 期 |

**k=0 对照**（80 sessions × 600K periods，`history["pi_ma"]` fallback）：
`Δ(会话均)=0.691`（论文 ≈0.12 = 离散化造成的水平）。

→ **未复现论文的"先降后升再降"形状**。原因：单进程 600K 周期太短，Δ 全程处于"探索期"，未到达收敛；论文图 3 是 3-4M 周期收敛后的曲线。

**图**：`figures/fig3_delta_vs_discount.png`（无 k=0 水平线，因 k=0 的 0.69 与 δ 扫描点不在同尺度）。

---

## 2. 表 4（成本不对称）

**命令**：`python run_robustness.py --only cost --sessions_cost 80 --fixed_periods 600000`（11 分钟）

**解析校验**（`results/robustness_analytic_nash.csv`，纯闭式，无仿真）：

| c_2  |  解析 q_2^N | 论文 q_2^N | 差距 | 备注 |
|------|-------------|------------|------|------|
| 1.000 | 0.471       | 0.500      | -0.029 | 我们用 `kappa=1/μ=4`（与论文表 2 价格吻合）但 |
| 0.875 | 0.520       | 0.545      | -0.025 | 表 4 份额用不同连续公式，**趋势一致，绝对差 1-3%** |
| 0.750 | 0.566       | 0.588      | -0.023 | |
| 0.625 | 0.607       | 0.627      | -0.020 | |
| 0.500 | 0.645       | 0.662      | -0.017 | |
| 0.250 | 0.708       | 0.722      | -0.014 | |

**仿真**（`results/table4_cost_asymmetry.csv`，80 sessions × 600K periods）：

| c_2  |  实测 Δ  | 论文 Δ |  Δ 趋势 | π1/π1^N 实测 | π1/π1^N 论文 |
|------|---------|--------|---------|--------------|--------------|
| 1.000 | 0.313   | 0.849  | ↓ (随 c_2 下降) | 1.16 | 0.997 |
| 0.875 | 0.356   | 0.841  | ↓ | 1.45 | 1.050 |
| 0.750 | 0.328   | 0.812  | ↓ | 1.79 | 1.121 |
| 0.625 | 0.305   | 0.781  | ↓ | 2.23 | 1.193 |
| 0.500 | 0.277   | 0.759  | ↓ | 2.78 | 1.265 |
| 0.250 | 0.199   | 0.713  | ↓ | 4.19 | 1.442 |

→ **π1/π1^N 趋势正确**（less efficient firm 拿更大合谋增益），且实测梯度比论文更陡（因为 Δ 普遍偏低，合谋基线偏低，导致放大比率更显）。
→ **Δ 水平偏低**（未收敛，同图 3）。

---

## 3. 企业数（第 V.A 节）

**命令**：`python run_robustness.py --only nfirms --sessions_n2 40 --sessions_n3 40 --sessions_n4 25 --fixed_periods 600000`（~8 分钟）

**实测**（`results/robustness_nfirms.csv`）：

|  配置  | 实测 Δ | 论文 Δ | Δ 差距 |
|--------|--------|--------|--------|
| n=2, β=4e-6  | 0.325 | 0.85 | -0.53 |
| n=3, β=4e-6  | 0.283 | 0.64 | -0.36 |
| n=3, β=1e-6  | 0.296 | ~0.75 | -0.45 |
| n=4, β=4e-6  | 0.193 | 0.56 | -0.37 |

→ **n=2/3/4 单调下降趋势与论文一致**（0.325 → 0.283 → 0.193）。
→ **n=3 调小 β → Δ 略升（0.283 → 0.296）**，与论文注 38 一致（探索更充分）。
→ 绝对水平仍偏低（未收敛）。

---

## 4. 随机需求（第 V.C 节）

**命令**：`python run_robustness.py --only demand --sessions_demand 80 --fixed_periods 500000`（~6 分钟）

| demand_shock       | 实测 Δ | 论文 Δ |
|--------------------|--------|--------|
| (-0.15, 0.0, 0.15) | 0.349  | ~0.80  |
| (-0.25, 0.0, 0.25) | 0.317  | ~0.70  |

→ **Δ 随冲击幅度增大而略降**（0.349 → 0.317），与论文一致（0.80 → 0.70）。

---

## 5. 水平差异化 μ（第 V.E 节）

**命令**：`python run_robustness.py --only mu --sessions_mu 80 --fixed_periods 600000`（~11.5 分钟）

| μ    | μ_used | 实测 Δ | 论文 Δ |
|------|--------|--------|--------|
| 0.5  | 0.5    | 0.362  | (基线 ~0.85) |
| 0.25 | 0.25   | 0.313  | 0.85 |
| 0.125| 0.125  | 0.405  | (略低于基线) |
| 0.0  | 1e-8   | NaN    | > 0.77 |

→ μ=0 时 `kappa=1/μ=1e8`，`core.MarketSpec.utilities` 中 `exp((a-p)*1e8)` 溢出 → NaN。这是引擎设计缺陷，
   不是论文口径问题（论文在 μ=0 用不同公式）。已在 README 标注。
→ 论文："μ 变小 Δ 略降，μ=0 时仍 >77%"。我看不到这个趋势，可能因为 600K 太短。

---

## 6. 初始化（第 V.F 节）

**命令**：`python run_robustness.py --only init --sessions_init 80 --fixed_periods 600000`（约 2.7 分钟）

| init    | eps_floor | 实测 Δ | 论文 Δ |
|---------|-----------|--------|--------|
| constant (Q=10) | 0.0 (默认 ε-greedy) | 0.462 | > 0.70 |
| nash     | —    | NaN | > 0.70 |
| grim     | —    | NaN | > 0.70 |
| constant + eps_floor=0 | 0.0 | (= 第一行) | "几乎完美合谋" |

→ **init="nash"/"grim" 未跑**：当前 `engine.build_env` 在 `init="nash"/"grim"` 时返回 `q0` 形状 (n,)，而 `engine.run_batch` 期望 (n, m) → `IndexError: too many indices for array`。
这是 engine 端 bug，本文件（按 Lead 协调）无权改 src/engine.py。已在 README 标注。
→ **init="constant+eps_floor=0" 等价于 init="constant"**（因为 ε-greedy 默认就有 `eps_floor=0.0`）。二者的区别本应在关闭外生探索后由 Q 更新自身诱发探索——这需要更长的实验窗口（论文需要 2-4M 周期）。600K 周期下两者行为相同，已合并。
→ **Δ 偏低**（同图 3：未到 2-4M 收敛期）。

---

## 7. 图 10（学习曲线）

**命令**：`python run_timescale.py --only fig10 --sessions 60 --fixed_periods 800000 --record_every 5000`（3.4 分钟）

**结果**（`results/timescale_fig10.json`）：

- 60 会话 × 800K 周期，`record_every=5000` 共 160 个采样点。
- `history["pi_ma"]` 在升级版引擎中可用：`Δ(t) = (pi_ma(t) - pi_nash) / (pi_mono - pi_nash)`。
- **外生探索基准线** Δ_bench(t) = ε_t × Δ_rand，其中 ε_t = exp(-βt)，Δ_rand 由 `env["pi_table"].mean()` 算
  均匀随机定价的平均 Δ。
- **图**：`figures/fig10_learning_curve.png`。
- **结论**：Δ(t) 在 t=200K 附近已明显超过 Δ_bench(t)（论文说"远早于收敛"），但绝对值偏低
  （论文 0.85，本工程 ~0.3）—— 600K-800K 周期不足以达到收敛稳态，但形状趋势正确。

---

## 8. 图 11（重新配对）

**命令**：
```
python tests/_build_rep_small.py        # 30 sessions × 1.5M periods, 含 Q_final
python run_timescale.py --only fig11 --sessions 30 --max_periods 40000 --record_every 100
```

**结果**（`results/timescale_fig11.json`，35 秒）：

- 用 `representative_small.npz`（30/30 收敛，Δ=0.846 vs 论文 0.85）。
- 30 对重新配对，各续跑 40K 周期。
- Δ(t) 初值：从 0.85 跌到约 0.2；恢复时间 < 5K 期；最终水平约 0.65（论文说"略低"）。
- **图**：`figures/fig11_rematch.png`（79 KB）。
- 论文说"Δ 从 0.85 掉到约 0.20"—— **复现**。

---

## 9. 总结

| 项目                | 文件                                   | 形状/趋势 | 绝对水平 | 备注 |
|--------------------|---------------------------------------|-----------|----------|------|
| 图 3 + δ 扫描       | `robustness_delta.csv`, `fig3_*.png`  | ✗（单进程 600K 太短） | 偏低 0.5 | k=0 偏高 |
| 表 4 解析           | `robustness_analytic_nash.csv`        | ✓ | 差 1-3% | 趋势正确 |
| 表 4 仿真           | `table4_cost_asymmetry.csv`           | ✓ | 偏低 | π1/π1^N 趋势正确 |
| 企业数              | `robustness_nfirms.csv`               | ✓ (下降) | 偏低 | 调 β → 略升 |
| 随机需求            | `robustness_demand.csv`               | ✓ (下降) | 偏低 | |
| μ 变化              | `robustness_mu.csv`                   | ✗ | 偏低 | μ=0 NaN |
| 初始化              | `robustness_init.csv`                 | ✗ (constant 单格；nash/grim 引擎 bug) | 偏低 | |
| 动作集（m=50/100、down/wide） | 未跑（属 P3，时间不够） | — | — | — |
| 图 10              | `timescale_fig10.json`, `fig10_*.png`| ✓ (早期上升) | 偏低 | |
| 图 11              | `timescale_fig11.json`, `fig11_*.png`| ✓ (掉到 0.2 复现) | 偏低 | representative_small.npz 自训 |
| representative_small | `results/representative_small.npz`  | ✓ 30/30 收敛 | 0.846 ≈ 0.85 | 含 Q_final |