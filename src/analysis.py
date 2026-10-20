"""analysis.py -- 极限策略的最优回应检验与 Q-loss（第 III.D 节）。

论文做法：给定对手的**极限策略**，本方的"真 Q 值"满足线性方程

        Q(s, a) = pi_i(s, a) + delta * max_{a'} Q(s', a'),
        s'      = 由 (本方动作 a, 对手按极限策略在 s 处的动作) 决定的下一状态，

求解后比较本方极限策略与最优回应，差额即 "Q-loss"（计算机科学中称 exploitation）。

本模块全部向量化到 (M, m) 矩阵上，用值迭代求解（delta<1 时唯一解，收敛很快）。
"""

from __future__ import annotations

import numpy as np

from core import profile_index


# --------------------------------------------------------------------------- #
#  状态转移与即时收益
# --------------------------------------------------------------------------- #
def _state_weights(n: int, m: int) -> np.ndarray:
    w = np.empty(n, dtype=np.int64)
    for j in range(n):
        w[j] = m ** (n - 1 - j)
    return w


def transition_and_reward(limit_j: np.ndarray, pi_tab: np.ndarray, m: int,
                          n: int = 2, i: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """给定对手 j 的极限策略 limit_j (M,)，返回 (T (M,m), R (M,m))。

    T[s, a] = 本方在状态 s 采取动作 a、对手按 limit_j[s] 行动时的下一状态
    R[s, a] = 该情形下本方的当期利润
    """
    M = limit_j.shape[0]
    w = _state_weights(n, m)
    aj = limit_j.astype(np.int64)
    j = 1 - i if n == 2 else None
    acts = np.zeros((M, m, n), dtype=np.int64)
    acts[:, :, j] = aj[:, None]                       # 对手动作固定
    acts[:, :, i] = np.arange(m)[None, :]             # 本方遍历全部动作
    T = acts @ w
    R = pi_tab[T, i]
    return T, R


def _prof_np(acts: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros(acts.shape[0], dtype=np.int64)
    for j in range(acts.shape[1]):
        out = out * m + acts[:, j]
    return out


def true_q(T: np.ndarray, R: np.ndarray, delta: float, tol: float = 1e-12,
           max_iter: int = 5000) -> np.ndarray:
    """值迭代求解 Q(s,a) = R(s,a) + delta * max_a' Q(T(s,a), a')。"""
    M, m = R.shape
    Q = np.zeros((M, m), dtype=float)
    for _ in range(max_iter):
        V = Q.max(axis=1)
        Qn = R + delta * V[T]
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn
            break
        Q = Qn
    return Q


# --------------------------------------------------------------------------- #
#  单个会话
# --------------------------------------------------------------------------- #
def session_best_response(limit_actions_row: np.ndarray, env: dict, delta: float,
                          onpath_row: np.ndarray | None = None,
                          tol: float = 1e-12) -> dict:
    """对一家企业给出：真 Q、最优回应、在轨状态上的偏离与 Q-loss。

    limit_actions_row: (n, M) int16
    env: build_env 的输出（需含 pi_table、m 可由 grid 推出）
    """
    grid = env["grid"]
    m = len(grid)
    n, M = limit_actions_row.shape
    pi_tab = env["pi_table"]
    out = {"Q": [], "best": [], "is_best_onpath": [], "qloss_onpath": [],
           "qloss_all": [], "is_best_all": []}
    for i in range(n):
        j = 1 - i if n == 2 else None
        T, R = transition_and_reward(limit_actions_row[j], pi_tab, m, n=n, i=i)
        Q = true_q(T, R, delta, tol=tol)
        best = np.argmax(Q, axis=1).astype(np.int16)
        V = Q.max(axis=1)
        aL = limit_actions_row[i]
        loss = np.where(V > 0, (V - Q[np.arange(M), aL]) / np.maximum(V, 1e-12), 0.0)
        if onpath_row is not None and onpath_row.size:
            op = np.asarray(onpath_row, dtype=np.int64)
            op = op[op >= 0]
            on = best[op] == aL[op]
            out["is_best_onpath"].append(bool(on.all()))
            out["qloss_onpath"].append(float(loss[op].mean()) if op.size else np.nan)
        else:
            out["is_best_onpath"].append(np.nan)
            out["qloss_onpath"].append(np.nan)
        out["is_best_all"].append(bool((best == aL).all()))
        out["qloss_all"].append(float(loss.mean()))
        out["Q"].append(Q)
        out["best"].append(best)
    return out


# --------------------------------------------------------------------------- #
#  跨会话向量化的最优回应检验（参数网格用）
# --------------------------------------------------------------------------- #
def evaluate_sessions_vec(limit_actions: np.ndarray, pi_tab: np.ndarray,
                          grid: np.ndarray, delta: float, onpath: np.ndarray,
                          tol: float = 1e-12, max_iter: int = 4000) -> dict:
    """一次算完 S 个会话的最优回应检验（第 III.D 节）。

    limit_actions: (S, n, M) int16；onpath: (S, W) int32（末端在轨状态）
    返回每个会话的: 是否在轨 Nash、是否至少一方最优、Q-loss(on-path/all)，
    以及 τ=2 惩罚强度（非偏离方相对事前价格的降价百分比，见图 7）。
    """
    S, n, M = limit_actions.shape
    m = len(grid)
    W = min(onpath.shape[1], 200)
    op = onpath[:, -W:].astype(np.int64)                      # (S, W)
    last = op[:, -1]
    rows = np.arange(S)[:, None]
    cols = np.arange(M)[None, :]
    w = _state_weights(n, m)

    out = {
        "nash_onpath": np.ones(S, dtype=bool),
        "either_best_onpath": np.zeros(S, dtype=bool),
        "nash_allstates": np.ones(S, dtype=bool),
        "qloss_onpath": np.zeros(S),
        "qloss_all": np.zeros(S),
        "punish_tau2": np.zeros(S),
    }
    pre_actions = limit_actions[rows[:, 0], :, last].astype(np.int64)   # (S, n) 事前动作
    pre_price = grid[pre_actions[:, 0]].astype(float)
    br_dev = {i: _static_br_all(pi_tab, m, n, i, pre_actions[:, 1 - i])
              for i in range(n)}

    for i in range(n):
        j = 1 - i if n == 2 else None
        aj = limit_actions[:, j, :].astype(np.int64)          # (S, M)
        acts = np.repeat(np.repeat(aj[:, :, None, None], m, axis=2), n, axis=3)
        acts[:, :, :, i] = np.arange(m)[None, None, :]      # (S, M, m, n)
        idx = _prof_np3(acts, m)                            # (S, M, m)
        T, R = idx, pi_tab[idx, i]
        Q = np.zeros((S, M, m))
        base = (np.arange(S) * M)[:, None, None]
        for _ in range(max_iter):
            Vm = Q.max(axis=2).reshape(S * M)
            Qn = R + delta * Vm[base + T]
            if np.max(np.abs(Qn - Q)) < tol:
                Q = Qn
                break
            Q = Qn
        best = Q.argmax(axis=2).astype(np.int16)
        V = Q.max(axis=2)
        aL = limit_actions[:, i, :].astype(np.int64)
        loss = np.where(V > 1e-12, (V - Q[rows, cols, aL]) / np.where(V > 1e-12, V, 1.0), 0.0)
        hit = (best == aL)
        on_hit = hit[rows, op]
        out["nash_onpath"] &= on_hit.all(axis=1)
        out["either_best_onpath"] |= on_hit.all(axis=1)
        out["nash_allstates"] &= hit.all(axis=1)
        out["qloss_onpath"] += loss[rows, op].mean(axis=1) / n
        out["qloss_all"] += loss.mean(axis=1) / n

        # τ=2 惩罚强度: 偏离方 τ=1 采用一阶最优回应, 非偏离方 τ=2 的定价
        prof1 = pre_actions.copy()
        prof1[:, i] = br_dev[i]
        s2 = prof1 @ w
        p2 = grid[limit_actions[:, j, :][rows, s2[:, None]]].ravel()
        out["punish_tau2"] += (p2.astype(float) / pre_price - 1.0) / n
    return out


def _prof_np3(acts: np.ndarray, m: int) -> np.ndarray:
    """(S, M, m, n) 动作张量的 base-m 编码 -> (S, M, m)。"""
    out = np.zeros(acts.shape[:3], dtype=np.int64)
    for c in range(acts.shape[3]):
        out = out * m + acts[:, :, :, c]
    return out


def _static_br_all(pi_tab: np.ndarray, m: int, n: int, i: int,
                   rival_actions: np.ndarray) -> np.ndarray:
    """对每个会话给定对手的事前动作, 返回本方的一阶最优回应动作。

    rival_actions: (S,) -> 返回 (S,)
    """
    r = np.asarray(rival_actions, dtype=np.int64)
    S = r.shape[0]
    j = 1 - i
    acts = np.zeros((S, m, n), dtype=np.int64)
    acts[:, :, j] = r[:, None]                     # 对手动作固定
    acts[:, :, i] = np.arange(m)[None, :]          # 本方遍历全部动作
    prof = np.zeros((S, m), dtype=np.int64)
    for c in range(n):
        prof = prof * m + acts[:, :, c]
    return pi_tab[prof, i].argmax(axis=1).astype(np.int64)


def evaluate_sessions(res, delta: float | None = None, onpath_window: int = 200,
                      max_sessions: int | None = None) -> dict:
    env = res.env
    n, M = res.limit_actions.shape[1], res.limit_actions.shape[2]
    delta = res.meta["delta"] if delta is None else delta
    R = res.limit_actions.shape[0] if max_sessions is None else min(max_sessions, res.limit_actions.shape[0])
    W = min(onpath_window, res.onpath_states.shape[1])

    both_on, either_on, ql_on, ql_all, both_all = [], [], [], [], []
    for r in range(R):
        op = res.onpath_states[r, -W:]
        d = session_best_response(res.limit_actions[r], env, delta, onpath_row=op)
        both_on.append(bool(np.all(d["is_best_onpath"])))
        either_on.append(bool(np.any(d["is_best_onpath"])))
        ql_on.append(np.mean(d["qloss_onpath"]))
        ql_all.append(np.mean(d["qloss_all"]))
        both_all.append(bool(np.all(d["is_best_all"])))

    return {
        "n_sessions": R,
        "freq_nash_onpath": float(np.mean(both_on)),
        "freq_either_best_onpath": float(np.mean(either_on)),
        "freq_nash_allstates": float(np.mean(both_all)),
        "qloss_onpath_mean": float(np.nanmean(ql_on)),
        "qloss_onpath_sd": float(np.nanstd(ql_on)),
        "qloss_all_mean": float(np.nanmean(ql_all)),
        "qloss_all_sd": float(np.nanstd(ql_all)),
        "is_best_onpath": np.array(both_on),
        "qloss_onpath": np.array(ql_on),
    }


# --------------------------------------------------------------------------- #
#  静态最优回应（偏离实验用）
# --------------------------------------------------------------------------- #
def static_best_response_action(pi_tab: np.ndarray, m: int, n: int, i: int,
                                rival_actions: np.ndarray) -> np.ndarray:
    """固定对手动作向量时, 本方在各状态下的静态最优回应动作 (M,)。"""
    r = np.asarray(rival_actions, dtype=np.int64)
    S = r.shape[0]
    j = 1 - i
    acts = np.zeros((S, m, n), dtype=np.int64)
    acts[:, :, j] = r[:, None]
    acts[:, :, i] = np.arange(m)[None, :]
    prof = np.zeros((S, m), dtype=np.int64)
    for c in range(n):
        prof = prof * m + acts[:, :, c]
    return pi_tab[prof, i].argmax(axis=1).astype(np.int64)


def best_response_to_price(pi_tab: np.ndarray, m: int, n: int, i: int,
                           rival_action: int) -> int:
    """对手固定在 rival_action 时, 本方的静态最优回应。"""
    j = 1 - i if n == 2 else None
    prof = np.zeros(n, dtype=np.int64)
    prof[j] = rival_action
    best_a, best_v = 0, -np.inf
    for a in range(m):
        prof[i] = a
        v = pi_tab[profile_index(prof, m), i]
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def deviation_gain(payoff_path: np.ndarray, baseline_path: np.ndarray,
                   delta: float) -> float:
    """偏离相对基线的贴现利润变化（百分比）。

    payoff_path / baseline_path: 从 tau=0 开始的逐期每企业利润。
    """
    T = min(len(payoff_path), len(baseline_path))
    d = 0.0
    w = 1.0
    for t in range(T):
        d += w * (payoff_path[t] - baseline_path[t])
        w *= delta
    base = 0.0
    w = 1.0
    for t in range(T):
        base += w * baseline_path[t]
        w *= delta
    return d / base if base != 0 else np.nan