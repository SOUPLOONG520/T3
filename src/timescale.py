"""timescale.py -- Calvano et al. (2020) 论文第 VI 节（A-B）时间尺度实验。

A. 图 10：学习曲线 Δ(t) + 外生探索基准线。
   在 t=0..T 上 record_every 间隔记录 pi；每个 t 的"外生探索基准线"为
       Delta_bench(t) = (1-eps_t) * 0 + eps_t * Delta_rand
   其中 Delta_rand = 均匀随机定价时的 Δ（用 env["pi_table"] 计算随机平均利润）；
   不探索时按 Bertrand-Nash 价（由 defect 近似）定价 → 该项为 0。
   eps_t = exp(-beta * t)。

B. 图 11：重新配对。
   复用代表性实验的收敛 Q 矩阵（rows=收敛会话），重新组合企业 1 (会话 i)
   与企业 2 (会话 j)，关闭探索，按式 (4) 续跑 Q-learning，记录 Δ(t)。
   若 results/representative.npz 不存在，本模块自带一次性 30 会话 representative
   重新跑并保存（small representative），后续续跑函数直接消费它。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from core import MarketSpec, benchmark_prices, profile_index
from engine import SimConfig
from parallel import DEFAULT_WORKERS, run_config


# --------------------------------------------------------------------------- #
# 图 10：学习曲线
# --------------------------------------------------------------------------- #
def run_learning_curve(
    n_sessions: int = 50,
    n_workers: int = 1,
    fixed_periods: int = 1_000_000,
    record_every: int = 5000,
    seed: int = 0,
) -> dict:
    """图 10：固定 1M 周期 × record_every=5000 记录 history["pi_ma"]。

    Lead 协调（2026-10-20 19:08）：单进程 + 60 会话 + 800K 周期（约 3 分钟）。
    若 history 不可用则用 res.pi_final fallback（按 profit_window 1000 平均）。
    """
    cfg = BaseConfigTC().to_sim(fixed_periods=fixed_periods,
                                 record_every=record_every)
    res = run_config(cfg, n_sessions, n_workers=n_workers)
    env = res.env
    pi_nash = env["pi_nash"]
    pi_mono = env["pi_mono"]
    pi_table = env["pi_table"]
    pi_random_per_firm = pi_table.mean(axis=0)
    delta_rand = float(np.mean(pi_random_per_firm - pi_nash) / (pi_mono - pi_nash))

    method = ""
    if res.history is not None and "pi_ma" in res.history:
        pi_ma = res.history["pi_ma"]
        eps = res.history["eps"]
        T = pi_ma.shape[1]
        time_axis = np.arange(T) * record_every
        delta_ma = (pi_ma - pi_nash) / (pi_mono - pi_nash)
        eps_t = eps[0]
        delta_bench = eps_t * delta_rand
        method = "history_pi_ma"
    else:
        # fallback：每个会话的最终 pi（即 _finalize 写出的 ring 1000 期均值）
        # 这是 fixed_periods 模式下 res.delta() 等价的口径
        pi_final = np.asarray(res.pi_final, dtype=float)
        delta_ma = (pi_final[:, None] - pi_nash) / (pi_mono - pi_nash)
        time_axis = np.array([fixed_periods])
        # 用 eps = exp(-beta * fixed_periods)
        eps_t_final = float(np.exp(-float(env.get("beta", 4e-6)) * fixed_periods))
        delta_bench = np.array([eps_t_final * delta_rand])
        method = "res.pi_final fallback"

    T_out = min(delta_ma.shape[1], delta_bench.shape[0])
    return {
        "time_axis": time_axis[:T_out].tolist(),
        "delta_mean": delta_ma.mean(axis=0)[:T_out].tolist(),
        "delta_std": delta_ma.std(axis=0)[:T_out].tolist(),
        "delta_bench": delta_bench[:T_out].tolist(),
        "delta_rand": float(delta_rand),
        "n_sessions": int(n_sessions),
        "fixed_periods": int(fixed_periods),
        "record_every": int(record_every),
        "window": 100,
        "pi_random_per_firm": pi_random_per_firm.tolist(),
        "pi_nash": float(pi_nash),
        "pi_mono": float(pi_mono),
        "method": method,
    }


# --------------------------------------------------------------------------- #
# 图 11：重新配对
# --------------------------------------------------------------------------- #
def run_rematch(
    npz_path: str | None = None,
    n_pairs: int = 30,
    max_periods: int = 40_000,
    record_every: int = 100,
    seed: int = 0,
    n_workers: int = DEFAULT_WORKERS,
) -> dict:
    """重新配对实验：把会话 i 的 firm-0 与会话 j (i≠j) 的 firm-1 组成新对。

    Lead 协调：图 11 用 representative_small.npz（30 会话，1.5M 周期，
    含 Q_final[session, agent, state, action]）。
    dev-graph 的 representative.npz 不含 Q，不能用于图 11。
    """
    if npz_path is None:
        npz_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "results", "representative_small.npz"
        )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} 不存在；先运行 tests/_build_rep_small.py 生成。"
        )
    data = np.load(npz_path, allow_pickle=True)
    if "Q_final" not in data.files:
        raise RuntimeError(
            f"{npz_path} 中没有 Q_final；请用 return_q=True 重新生成。"
        )
    Q_all = data["Q_final"]
    # 重新配对：从 converged 的会话里随机抽
    rng = np.random.default_rng(seed)
    conv_idx = np.where(data["converged"])[0]
    if len(conv_idx) < 2:
        raise RuntimeError("Not enough converged sessions in representative.npz")
    pairs = []
    while len(pairs) < n_pairs:
        i, j = rng.choice(conv_idx, 2, replace=False)
        pairs.append((int(i), int(j)))

    # 构造每对的续跑输入
    pi_tab = data["pi_table"]
    delta = float(data["delta"])
    m = int(data["m"])
    n = int(data["n"])
    M = int(data["M"])
    pi_nash = float(data["pi_nash"])
    pi_mono = float(data["pi_mono"])

    if "Q_final" not in data.files:
        raise RuntimeError(
            f"{npz_path} 中没有 Q_final；请用 return_q=True 重新生成。"
        )
    Q_all = data["Q_final"]              # (N, n, M, m)

    T_rec = max_periods // record_every + 1
    pi_trace = np.zeros((n_pairs, T_rec))

    for p_idx, (i, j) in enumerate(pairs):
        Q0 = np.array(Q_all[i, 0], dtype=np.float64).copy()
        Q1 = np.array(Q_all[j, 1], dtype=np.float64).copy()
        best0 = Q0.argmax(axis=1).astype(np.int16)
        best1 = Q1.argmax(axis=1).astype(np.int16)
        s = int(rng.integers(0, M))
        W = 100
        pi_ring = np.zeros(W)
        ptr = 0
        cnt = 0
        delta_hist = np.zeros(T_rec)

        for t in range(max_periods + 1):
            a0 = int(best0[s])
            a1 = int(best1[s])
            idx = profile_index([a0, a1], m)
            pi_pair = pi_tab[idx]
            pi_mean = float(pi_pair.mean())
            if t % record_every == 0:
                slot = t // record_every
                if slot < T_rec:
                    pi_avg = float(pi_ring[:cnt].mean()) if cnt > 0 else pi_mean
                    delta_hist[slot] = (pi_avg - pi_nash) / (pi_mono - pi_nash)
            pi_ring[ptr] = pi_mean
            ptr = (ptr + 1) % W
            cnt = min(cnt + 1, W)
            ns = a0 * m + a1
            alpha = 0.15
            tgt0 = float(Q0[ns].max())
            tgt1 = float(Q1[ns].max())
            Q0[s, a0] = (1 - alpha) * Q0[s, a0] + alpha * (pi_pair[0] + delta * tgt0)
            Q1[s, a1] = (1 - alpha) * Q1[s, a1] + alpha * (pi_pair[1] + delta * tgt1)
            new_best0 = int(Q0[s].argmax())
            new_best1 = int(Q1[s].argmax())
            best0[s] = new_best0
            best1[s] = new_best1
            s = ns

        pi_trace[p_idx] = delta_hist

    return {
        "time_axis": (np.arange(T_rec) * record_every).tolist(),
        "delta_traces": pi_trace.tolist(),
        "n_pairs": int(n_pairs),
        "max_periods": int(max_periods),
        "record_every": int(record_every),
        "window": 100,
        "delta_mean": pi_trace.mean(axis=0).tolist(),
        "delta_std": pi_trace.std(axis=0).tolist(),
    }


# --------------------------------------------------------------------------- #
# 自包含的 small representative（含 Q 矩阵）
# --------------------------------------------------------------------------- #
def _ensure_representative_npz(npz_path: str, n_workers: int = DEFAULT_WORKERS,
                                n_sessions: int = 30,
                                max_periods: int = 2_000_000) -> dict:
    """占位 — Lead 协调说明：本函数不再自己生成 representative.npz。

    representative.npz（含 Q 矩阵）由队友 dev-graph 提供。
    若调用本函数而 npz 不存在，直接报错，让 run_rematch 也失败（被阻塞）。
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} 不存在；按 Lead 协调说明，representative.npz 必须由 "
            "dev-graph 生成，本模块不再自己造。"
        )
    return np.load(npz_path, allow_pickle=True)


def _train_representative_with_q(npz_path: str, n_sessions: int,
                                   max_periods: int,
                                   n_workers: int) -> dict:
    """占位 — 已被 _ensure_representative_npz 取代。保留以防旧脚本引用。"""
    raise RuntimeError(
        "_train_representative_with_q 已废弃；按 Lead 协调说明，请使用 "
        "dev-graph 生成的 representative.npz（results/representative.npz）。"
    )


# --------------------------------------------------------------------------- #
# 基线配置
# --------------------------------------------------------------------------- #
@dataclass
class BaseConfigTC:
    alpha: float = 0.15
    beta: float = 4e-6
    patience: int = 100_000
    max_periods: int = 5_000_000
    profit_window: int = 1000
    fixed_periods: int = 1_000_000
    record_every: int = 5000

    def to_sim(self, **overrides) -> SimConfig:
        kwargs = asdict(self)
        kwargs.update(overrides)
        kwargs.pop("to_sim", None)
        return SimConfig(**kwargs)


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def save_json(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)