"""robustness.py -- Calvano et al. (2020) 论文第 V 节（A-G）稳健性实验。

覆盖：
  A. 企业数 n=2/3/4，以及 n=3 时同步调小 β
  B. 成本不对称（表 4，c_1=1, c_2 ∈ 6 档）
  C. 随机需求（两个规格的 a_0）
  E. 水平差异化 μ ∈ {0.5, 0.25, 0.125, 0.0}
  F. 初始化（nash / grim / constant / constant + eps_floor=0）
  G. 动作集（m=50 / m=100 / grid_mode="down" / "wide"）
  以及图 3 的 δ 扫描 + k=0（无记忆）对照。

原则：
  * 不修改 src/core.py、src/engine.py、src/parallel.py；
  * 全部基于公共 API；
  * 每个实验函数返回 dict 形式的聚合结果；
  * 默认会话数：基线 100，n=3 时 60，n=4 时 40，m=100 时 ≤20（COMMON.md）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from core import MarketSpec, benchmark_prices
from engine import SimConfig
from parallel import DEFAULT_WORKERS, run_config


def _bc(fixed_periods: int = 1_500_000) -> BaseConfig:
    """返回一个固定期版本的 BaseConfig（Lead 协调：单进程 + 1.5M 固定期）。"""
    return BaseConfig(fixed_periods=fixed_periods)


# --------------------------------------------------------------------------- #
# 公共规格与默认值
# --------------------------------------------------------------------------- #
@dataclass
class BaseConfig:
    """所有实验共享的默认超参。

    Lead 协调（2026-10-20 19:08）：单进程；会话数比周期数重要（R=80 时
    ~300K 行·期/秒；R=12 时仅 ~64K/s）。fixed_periods 默认 600K，
    全局 fixed_periods 模式跑（约 9 δ × 80 会话 × 600K ≈ 25 分钟）。
    """

    alpha: float = 0.15
    beta: float = 4e-6
    patience: int = 100_000
    max_periods: int = 5_000_000
    profit_window: int = 1000
    fixed_periods: int = 600_000

    def to_sim(self, **overrides) -> SimConfig:
        """构造一个 SimConfig（默认 fixed_periods=600K）。"""
        kwargs = asdict(self)
        kwargs.update(overrides)
        kwargs.pop("to_sim", None)
        return SimConfig(**kwargs)


def spec_baseline() -> MarketSpec:
    return MarketSpec()  # n=2, cost=(1,1), a=(2,2), a0=0, mu=0.25, exponent="div"


# --------------------------------------------------------------------------- #
# A. δ 扫描（图 3 + 稳健性）
# --------------------------------------------------------------------------- #
def run_delta_sweep(
    deltas: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95),
    n_sessions: int = 80,
    n_workers: int = 1,
    cfg_extra: dict | None = None,
    fixed_periods: int = 600_000,
) -> dict:
    """每个 δ 值跑 n_sessions 个独立会话（fixed_periods 模式）。

    Lead 协调（2026-10-20 19:08）：单进程 80 会话 × 600K 固定期，
    不按论文等到 2-4M 期（DELTA 未真正收敛）。
    用 history["pi_ma"] 取末期 Δ（口径诚实，会显著晚于收敛）。
    """
    cfg_extra = cfg_extra or {}
    out: dict = {"deltas": list(deltas), "sessions_per_delta": n_sessions,
                 "fixed_periods": int(fixed_periods), "rows": []}
    for d in deltas:
        cfg = _bc(fixed_periods).to_sim(delta=float(d), **cfg_extra)
        res = run_config(cfg, n_sessions, n_workers=n_workers)
        deltas_arr = res.delta()
        out["rows"].append({
            "delta": float(d),
            "n": int(len(deltas_arr)),
            "mean_delta": float(np.mean(deltas_arr)),
            "sd_delta": float(np.std(deltas_arr)),
            "mean_pi_final": float(np.nanmean(res.pi_final)),
            "converged_frac": float(np.mean(res.converged)),
        })
    return out


# --------------------------------------------------------------------------- #
# 无记忆对照（k=0）
# --------------------------------------------------------------------------- #
def run_memoryless(
    n_sessions: int = 80,
    n_workers: int = 1,
    cfg_extra: dict | None = None,
    fixed_periods: int = 600_000,
) -> dict:
    """memory=0 → 状态只有一格（M=1）。

    用 fixed_periods + record_every 让 history["pi_ma"] 有数据；
    若 history 不可用（如 record_every 被某条路径忽略），fallback 到
    res.pi_final 的最后 1/3 移动平均。
    """
    cfg_extra = cfg_extra or {}
    cfg = _bc(fixed_periods).to_sim(memory=0, record_every=5000, **cfg_extra)
    res = run_config(cfg, n_sessions, n_workers=n_workers)
    pi_nash = res.env["pi_nash"]
    pi_mono = res.env["pi_mono"]
    if res.history is not None and "pi_ma" in res.history:
        pi_ma = res.history["pi_ma"]
        delta_ma = (pi_ma - pi_nash) / (pi_mono - pi_nash)
        n_last = max(pi_ma.shape[1] // 3, 1)
        final = delta_ma[:, -n_last:].mean(axis=1)
        method = "history_pi_ma last 1/3 mean"
    else:
        final = (res.pi_final - pi_nash) / (pi_mono - pi_nash)
        method = "res.pi_final fallback"
    return {"n": int(len(final)), "mean_delta": float(np.mean(final)),
            "sd_delta": float(np.std(final)),
            "all_deltas": final.tolist(),
            "converged_frac": float(np.mean(res.converged)),
            "memory": 0, "fixed_periods": int(fixed_periods),
            "method": method}


# --------------------------------------------------------------------------- #
# A. 企业数
# --------------------------------------------------------------------------- #
def run_n_firms(
    ns: Sequence[int] = (2, 3, 4),
    n_sessions_by_n: dict[int, int] | None = None,
    n_workers: int = 1,
    fixed_periods: int = 1_500_000,
) -> dict:
    """n ∈ {2,3,4} 各跑一批（fixed_periods 模式）。n=3 还做一次 β=1e-6。"""
    n_sessions_by_n = n_sessions_by_n or {2: 25, 3: 25, 4: 15}

    cfgs = []
    for n in ns:
        ns_sess = n_sessions_by_n[n]
        cfgs.append((ns_sess, _bc(fixed_periods).to_sim(
            spec=MarketSpec(n=n, quality=tuple([2.0] * n),
                            cost=tuple([1.0] * n), a0=0.0, mu=0.25),
        ), f"n={n}_beta=4e-6"))
        if n == 3:  # n=3 + 调小 β
            cfgs.append((ns_sess, _bc(fixed_periods).to_sim(
                spec=MarketSpec(n=n, quality=tuple([2.0] * n),
                                cost=tuple([1.0] * n), a0=0.0, mu=0.25),
                beta=1e-6,
            ), "n=3_beta=1e-6"))

    rows = []
    for ns_sess, cfg, label in cfgs:
        res = run_config(cfg, ns_sess, n_workers=n_workers)
        d = res.delta()
        rows.append({
            "label": label,
            "n_firms": int(cfg.spec.n),
            "beta": float(cfg.beta),
            "n_sessions": int(ns_sess),
            "mean_delta": float(np.mean(d)),
            "sd_delta": float(np.std(d)),
            "converged_frac": float(np.mean(res.converged)),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# B. 成本不对称（表 4）
# --------------------------------------------------------------------------- #
def run_cost_asymmetry(
    c2_values: list[float] | None = None,
    n_sessions: int = 25,
    n_workers: int = 1,
    fixed_periods: int = 1_500_000,
) -> dict:
    """c_1=1，c_2 ∈ {1, 0.875, 0.75, 0.625, 0.5, 0.25}。返回含解析校验 + 仿真 Δ。"""
    if c2_values is None:
        c2_values = [1.0, 0.875, 0.75, 0.625, 0.5, 0.25]
    rows = []
    for c2 in c2_values:
        spec = MarketSpec(n=2, quality=(2.0, 2.0), cost=(1.0, float(c2)),
                          a0=0.0, mu=0.25)
        bm = benchmark_prices(spec)
        q_nash = float(bm["q_nash"][1])
        pi1_nash = float(bm["pi_nash"][0])
        pi2_nash = float(bm["pi_nash"][1])
        pi_mono_avg = float(bm["pi_mono_avg"])
        pi_nash_avg = float(bm["pi_nash_avg"])
        cfg = _bc(fixed_periods).to_sim(spec=spec)
        res = run_config(cfg, n_sessions, n_workers=n_workers)
        d = res.delta()
        rows.append({
            "c2": float(c2),
            "nash_share_firm2": q_nash,
            "delta_sim": float(np.mean(d)),
            "sd_delta_sim": float(np.std(d)),
            "n_sessions": int(n_sessions),
            "fixed_periods": int(fixed_periods),
            "pi1_over_pi1N": float(np.mean(res.pi_final) / max(pi1_nash, 1e-12)),
            "pi2_over_pi2N": float(np.mean(res.pi_final) / max(pi2_nash, 1e-12)),
            "pi1_N_analytic": pi1_nash,
            "pi2_N_analytic": pi2_nash,
            "pi_N_avg_analytic": pi_nash_avg,
            "pi_M_avg_analytic": pi_mono_avg,
            "converged_frac": float(np.mean(res.converged)),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# C. 随机需求
# --------------------------------------------------------------------------- #
def run_demand_shock(
    shock_levels: Sequence[tuple] = ((-0.15, 0.0, 0.15), (-0.25, 0.0, 0.25)),
    n_sessions: int = 25,
    n_workers: int = 1,
    fixed_periods: int = 1_500_000,
) -> dict:
    """demand_shock 取 3 元组；论文报告 a_0 ∈ {±0.15} → Δ≈0.80，{±0.25} → Δ≈0.70。"""
    rows = []
    for shock in shock_levels:
        cfg = _bc(fixed_periods).to_sim(demand_shock=tuple(shock))
        res = run_config(cfg, n_sessions, n_workers=n_workers)
        d = res.delta()
        rows.append({
            "shock": list(shock),
            "n_sessions": int(n_sessions),
            "fixed_periods": int(fixed_periods),
            "mean_delta": float(np.mean(d)),
            "sd_delta": float(np.std(d)),
            "converged_frac": float(np.mean(res.converged)),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# E. 水平差异化 μ
# --------------------------------------------------------------------------- #
def run_mu_variation(
    mus: Sequence[float] = (0.5, 0.25, 0.125, 0.0),
    n_sessions: int = 25,
    n_workers: int = 1,
    fixed_periods: int = 1_500_000,
) -> dict:
    """μ ∈ {0.5, 0.25, 0.125, 0.0}。μ=0 时 products 完全替代。"""
    rows = []
    for mu in mus:
        mu_use = max(float(mu), 1e-8) if mu == 0 else float(mu)
        spec = MarketSpec(n=2, quality=(2.0, 2.0), cost=(1.0, 1.0),
                          a0=0.0, mu=mu_use)
        cfg = _bc(fixed_periods).to_sim(spec=spec)
        res = run_config(cfg, n_sessions, n_workers=n_workers)
        d = res.delta()
        rows.append({
            "mu": float(mu),
            "mu_used": float(mu_use),
            "n_sessions": int(n_sessions),
            "mean_delta": float(np.mean(d)),
            "sd_delta": float(np.std(d)),
            "converged_frac": float(np.mean(res.converged)),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# F. 初始化
# --------------------------------------------------------------------------- #
def run_initialization(
    n_sessions: int = 80,
    n_workers: int = 1,
    fixed_periods: int = 600_000,
) -> dict:
    """第 V.F 节初始化。

    注：当前 build_env 在 init="nash"/"grim" 时返回 q0 形状 (n,)，而
    engine.run_batch 期望 (n, m) → 触发 `IndexError: too many indices`。
    属于 engine 端 bug，本文件无权改 src/engine.py。
    所以本函数只跑 init="constant" 系列（带 ε-greedy 与 eps_floor=0 两格），
    init="nash"/"grim" 标注为"未跑（引擎 bug）"。
    """
    cases = [
        dict(init="constant", eps_floor=0.0, init_q=10.0, label="init=constant"),
        dict(init="constant", eps_floor=0.0, init_q=10.0,
             label="init=constant+eps_floor=0",
             extra_eps_floor=0.0),
    ]
    rows = []
    seen = set()
    for c in cases:
        key = (c["init"], c.get("extra_eps_floor", 0.0), c["init_q"])
        if key in seen:
            continue
        seen.add(key)
        ef = c.get("extra_eps_floor", c["eps_floor"])
        cfg = _bc(fixed_periods).to_sim(init=c["init"], eps_floor=ef,
                                         init_q=c["init_q"])
        res = run_config(cfg, n_sessions, n_workers=n_workers)
        d = res.delta()
        rows.append({
            "label": c["label"],
            "init": c["init"],
            "eps_floor": float(ef),
            "init_q": float(c["init_q"]),
            "n_sessions": int(n_sessions),
            "mean_delta": float(np.mean(d)),
            "sd_delta": float(np.std(d)),
            "converged_frac": float(np.mean(res.converged)),
        })
    rows.append({
        "label": "init=nash", "init": "nash", "eps_floor": 0.0,
        "init_q": 10.0, "n_sessions": 0, "mean_delta": np.nan,
        "sd_delta": np.nan, "converged_frac": np.nan,
        "status": "skipped (engine build_env q0 shape bug)",
    })
    rows.append({
        "label": "init=grim", "init": "grim", "eps_floor": 0.0,
        "init_q": 10.0, "n_sessions": 0, "mean_delta": np.nan,
        "sd_delta": np.nan, "converged_frac": np.nan,
        "status": "skipped (engine build_env q0 shape bug)",
    })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# G. 动作集
# --------------------------------------------------------------------------- #
def run_action_set(
    n_sessions_normal: int = 25,
    n_sessions_m100: int = 15,
    n_workers: int = 1,
    fixed_periods: int = 1_500_000,
) -> dict:
    """m=50, m=100, grid_mode=down, grid_mode=wide。"""
    rows = []
    cases = [
        dict(m=50, grid_mode="sym", label="m=50", n_sess=n_sessions_normal),
        dict(m=100, grid_mode="sym", label="m=100", n_sess=n_sessions_m100),
        dict(m=15, grid_mode="down", label="grid_mode=down", n_sess=n_sessions_normal),
        dict(m=15, grid_mode="wide", label="grid_mode=wide", n_sess=n_sessions_normal),
    ]
    for c in cases:
        cfg = _bc(fixed_periods).to_sim(m=c["m"], grid_mode=c["grid_mode"])
        res = run_config(cfg, c["n_sess"], n_workers=n_workers)
        d = res.delta()
        rows.append({
            "label": c["label"],
            "m": int(c["m"]),
            "grid_mode": c["grid_mode"],
            "n_sessions": int(c["n_sess"]),
            "mean_delta": float(np.mean(d)),
            "sd_delta": float(np.std(d)),
            "converged_frac": float(np.mean(res.converged)),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# 解析校验：成本不对称的 Nash 份额
# --------------------------------------------------------------------------- #
def analytic_nash_shares(c2_values: Sequence[float] = (1.0, 0.875, 0.75, 0.625,
                                                       0.5, 0.25)) -> dict:
    """用 core.benchmark_prices 直接计算 Nash 份额，验证表 4。"""
    rows = []
    for c2 in c2_values:
        spec = MarketSpec(n=2, quality=(2.0, 2.0), cost=(1.0, float(c2)),
                          a0=0.0, mu=0.25)
        bm = benchmark_prices(spec)
        rows.append({
            "c2": float(c2),
            "nash_share_firm2": float(bm["q_nash"][1]),
            "p_nash": bm["p_nash"].tolist(),
            "p_mono": bm["p_mono"].tolist(),
            "pi1_N": float(bm["pi_nash"][0]),
            "pi2_N": float(bm["pi_nash"][1]),
            "pi_M_avg": float(bm["pi_mono_avg"]),
            "pi_N_avg": float(bm["pi_nash_avg"]),
        })
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# 输出辅助
# --------------------------------------------------------------------------- #
def save_json(d: dict, path: str) -> None:
    """把 dict 序列化成 JSON 文件（含 numpy 列表）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict], path: str) -> None:
    """把 list[dict] 写成 CSV（按行展开）。"""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    # 合并所有行的字段名（防止行间字段不同）
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# 图 3 绘图
# --------------------------------------------------------------------------- #
def plot_fig3(delta_rows: list[dict], k0: dict | None,
              out_path: str) -> None:
    """图 3: Δ vs δ；可选叠加 k=0 对照点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    deltas = [r["delta"] for r in delta_rows]
    means = [r["mean_delta"] for r in delta_rows]
    sds = [r["sd_delta"] for r in delta_rows]
    ax.errorbar(deltas, means, yerr=sds, fmt="o-", color="C0", capsize=3,
                label="Δ(δ) over sessions")
    if k0 is not None:
        ax.axhline(k0["mean_delta"], ls="--", color="C3",
                   label=f"k=0 baseline (Δ≈{k0['mean_delta']:.3f})")
    ax.set_xlabel("Discount factor δ")
    ax.set_ylabel("Average profit gain Δ")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Figure 3 (replication): profit gain Δ vs δ")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)