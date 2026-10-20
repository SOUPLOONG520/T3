"""engine.py -- 向量化 Q-learning 引擎 (重复博弈 Bertrand 寡头)。

复现 Calvano et al. (2020, AER 110(10)) 第 I 节 (Q-learning) 与第 II 节 (实验设计):

  * 状态  s_t = (p_{t-1}, ..., p_{t-k})   有界记忆 k, 式 (6)
  * 动作  A = m 个等距价格, 第 II.B 节
  * 收益  pi_i = (p_i - c_i) q_i, logit 需求, 式 (5)
  * 更新  Q_{t+1}(s,a) = (1-a)Q_t(s,a) + a[pi_t + delta*max_{a'} Q_t(s',a')], 式 (4)
  * 探索  eps_t = exp(-beta t), 式 (7)
  * 初始  Q_0 = pi_bar/(1-delta), 式 (8)
  * 收敛  argmax 策略映射连续 `patience` 期不变 (第 III.B 节), 硬上限 max_periods

实现要点 (性能):
  1. 把 (实验点 x 会话) 拉平成一个行维度, 全部用 numpy 一次算完;
  2. 已收敛的行会从活跃集合中剔除 ("行压缩"), 后续周期不再计算;
  3. 收敛瞬间即冻结该会话的 pi-bar 与在轨状态, 与论文 "到点即停" 一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from core import MarketSpec, benchmark_prices, init_q_values, payoff_table, price_grid

# 周期编码
CYCLE_CONST_SYM, CYCLE_CONST_ASYM, CYCLE_2, CYCLE_GE3, CYCLE_NOTCONV = 0, 1, 2, 3, 4
CYCLE_NAMES = {
    CYCLE_CONST_SYM: "1-Sym.",
    CYCLE_CONST_ASYM: "1-Asym.",
    CYCLE_2: "2",
    CYCLE_GE3: ">=3",
    CYCLE_NOTCONV: "not-converged",
}
CYCLE_ORDER = ["1-Sym.", "1-Asym.", "2", ">=3", "not-converged"]


# --------------------------------------------------------------------------- #
#  配置
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """一个实验点 (alpha, beta, ... ) 的完整设定。"""

    spec: MarketSpec = field(default_factory=MarketSpec)
    m: int = 15
    xi: float = 0.1
    grid_mode: str = "sym"            # "sym" | "down" | "wide"
    delta: float = 0.95
    memory: int = 1                   # k
    alpha: float = 0.15
    beta: float = 4e-6
    patience: int = 100_000           # 收敛判据: argmax 连续不变期数
    max_periods: int = 5_000_000      # 硬上限 (论文 1e9)
    fixed_periods: Optional[int] = None   # 定长运行 (学习曲线用), 忽略收敛
    profit_window: int = 1_000        # 计算 pi-bar 的窗口
    exploration: str = "epsilon"      # "epsilon" | "boltzmann"
    temperature: float = 1.0
    eps_floor: float = 0.0
    init: str = "random_rival"        # "random_rival" | "nash" | "grim" | "constant"
    init_q: float = 10.0
    demand_shock: Optional[tuple] = None
    record_every: int = 0             # >0: 记录学习曲线 (Figure 10)
    dtype: str = "float64"
    label: str = ""

    def key(self) -> str:
        return self.label or f"a{self.alpha:g}_b{self.beta:g}_d{self.delta:g}"


def env_signature(cfg: SimConfig) -> tuple:
    """用于判断一批 config 是否可共享同一套离散收益表。"""
    return (cfg.spec.n, tuple(cfg.spec.quality), tuple(cfg.spec.cost), cfg.spec.a0,
            cfg.spec.mu, cfg.spec.exponent, cfg.m, cfg.xi, cfg.grid_mode, cfg.init,
            cfg.init_q)


# --------------------------------------------------------------------------- #
#  环境预计算
# --------------------------------------------------------------------------- #
def build_env(cfg: SimConfig) -> dict:
    """基准价格 p^N / p^M、价格网格、离散收益表、初始 Q 值。"""
    bm = benchmark_prices(cfg.spec)
    pN = float(np.mean(bm["p_nash"]))
    pM = float(np.mean(bm["p_mono"]))
    grid = price_grid(pN, pM, m=cfg.m, xi=cfg.xi, mode=cfg.grid_mode)
    pi_tab = payoff_table(cfg.spec, grid)
    q0 = init_q_values(cfg.spec, pi_tab, cfg.m, cfg.delta)
    n = cfg.spec.n

    if cfg.init == "nash":
        a_nash = int(np.argmin(np.abs(grid - pN)))
        col = np.array([pi_tab[_prof([a_nash] * n, cfg.m), i] / (1.0 - cfg.delta)
                        for i in range(n)])
        q0 = np.repeat(col[:, None], cfg.m, axis=1)          # (n, m): 对手恒定 Nash 价
    elif cfg.init == "grim":
        a_coop = int(np.argmin(np.abs(grid - pM)))
        a_nash = int(np.argmin(np.abs(grid - pN)))
        coop = np.array([pi_tab[_prof([a_coop] * n, cfg.m), i] / (1.0 - cfg.delta)
                         for i in range(n)])
        pun = np.array([pi_tab[_prof([a_nash] * n, cfg.m), i] / (1.0 - cfg.delta)
                        for i in range(n)])
        q0 = np.tile(pun[:, None], (1, cfg.m))               # 默认惩罚
        for i in range(n):
            q0[i, a_coop] = coop[i]                          # 协同价给协同收益
    elif cfg.init == "constant":
        q0 = np.full((n, cfg.m), float(cfg.init_q))
    elif cfg.init != "random_rival":
        raise ValueError(f"unknown init {cfg.init}")

    return {
        "benchmarks": bm, "grid": grid, "pi_table": pi_tab, "q0": q0,
        "p_nash": pN, "p_mono": pM,
        "pi_nash": float(np.mean(bm["pi_nash"])),
        "pi_mono": float(np.mean(bm["pi_mono"])),
        "pi_nash_per_firm": bm["pi_nash"].copy(),
        "pi_mono_per_firm": bm["pi_mono"].copy(),
        "q_nash": bm["q_nash"].copy(),
    }


def _prof(actions: Sequence[int], m: int) -> int:
    idx = 0
    for a in actions:
        idx = idx * m + int(a)
    return idx


# --------------------------------------------------------------------------- #
#  结果容器
# --------------------------------------------------------------------------- #
@dataclass
class SessionResult:
    converged: np.ndarray             # (N,) bool
    t_conv: np.ndarray                # (N,) 收敛期数 (未收敛 -1)
    pi_final: np.ndarray              # (N,) 收敛时每企业平均利润
    longrun_actions: np.ndarray       # (N, n)
    longrun_prices: np.ndarray        # (N, n)
    cycle_type: np.ndarray            # (N,)
    limit_actions: np.ndarray         # (N, n, M) int16 极限策略
    onpath_states: np.ndarray         # (N, W) int32
    env: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    history: Optional[dict] = None
    Q_final: Optional[np.ndarray] = None      # (N, n, M, m) 仅当 run_batch(return_q=True)

    def delta(self) -> np.ndarray:
        """式 (9): 归一化超额利润 Delta。"""
        env = self.env
        return (self.pi_final - env["pi_nash"]) / (env["pi_mono"] - env["pi_nash"])

    def subset(self, idx) -> "SessionResult":
        idx = np.asarray(idx)
        return SessionResult(
            converged=self.converged[idx], t_conv=self.t_conv[idx],
            pi_final=self.pi_final[idx], longrun_actions=self.longrun_actions[idx],
            longrun_prices=self.longrun_prices[idx], cycle_type=self.cycle_type[idx],
            limit_actions=self.limit_actions[idx], onpath_states=self.onpath_states[idx],
            env=self.env, meta=self.meta, history=self.history,
            Q_final=None if self.Q_final is None else self.Q_final[idx])


# --------------------------------------------------------------------------- #
#  主体
# --------------------------------------------------------------------------- #
def run_batch(cfgs: Sequence[SimConfig], sessions: int, seed: int = 0,
              env: Optional[dict] = None, check_every: int = 100,
              return_q: bool = False) -> SessionResult:
    """在同一环境下并行跑 len(cfgs) x sessions 个独立会话。

    同一批内的 config 必须共享环境 (只有 alpha/beta 等学习参数可以不同)。

    实现: 把 (会话, 企业) 摊平成 "槽位" 维度 (slot = row * n + agent), 使内层
    每周期只需约 30 次 numpy 调用; 每 check_every 期检查一次收敛判据 (argmax
    映射是否变化), 变化则计数清零, 未变化则按 check_every 累加。
    """
    cfgs = list(cfgs)
    sig0 = env_signature(cfgs[0])
    for c in cfgs[1:]:
        if env_signature(c) != sig0:
            raise ValueError("run_batch 内所有 config 必须共享同一环境")
    env = env if env is not None else build_env(cfgs[0])

    spec = cfgs[0].spec
    n, m = spec.n, cfgs[0].m
    k = cfgs[0].memory
    M = (m ** (n * k)) if k > 0 else 1
    grid = env["grid"]
    pi_tab = env["pi_table"].astype(np.dtype(cfgs[0].dtype))
    P, R = len(cfgs), int(sessions)
    N = P * R
    delta = float(cfgs[0].delta)
    W = max(int(cfgs[0].profit_window), 1)
    fixed = bool(cfgs[0].fixed_periods)

    alpha_row = np.repeat(np.array([c.alpha for c in cfgs]), R)
    alpha_all = np.repeat(alpha_row, n)                    # 槽位顺序 (n*N,)
    om_all = 1.0 - alpha_all
    ad_all = alpha_all * delta
    pat_row = np.repeat(np.array([c.patience for c in cfgs], dtype=np.int64), R)
    limit_row = np.repeat(np.array([c.fixed_periods or c.max_periods for c in cfgs],
                                   dtype=np.int64), R)
    rng = np.random.default_rng(seed)
    rows_all = np.arange(N)
    agent_idx = np.arange(n)

    # ---- 槽位化数组 ---- #
    q0_slot = np.repeat(env["q0"].astype(cfgs[0].dtype), N, axis=0)     # (n*N, m)
    Q = np.repeat(q0_slot[:, None, :], M, axis=1)                       # (n*N, M, m)
    best = np.broadcast_to(np.argmax(q0_slot, axis=1)[:, None],
                           (n * N, M)).astype(np.int16).copy()
    stable = np.zeros(n * N, dtype=np.int64)
    beta_row = np.repeat(np.array([c.beta for c in cfgs]), R)

    if k > 0:
        hist = rng.integers(0, m, size=(N, k, n)).astype(np.int64)
        state = _encode_hist(hist, m)
    else:
        hist = np.zeros((N, 0, n), dtype=np.int64)
        state = np.zeros(N, dtype=np.int64)

    w_next = _state_weights(n, m)
    pi_ring = np.zeros((N, W))
    st_ring = np.zeros((N, W), dtype=np.int64)
    ptr = cnt = 0
    pi_out = np.full(N, np.nan)
    state_out = np.zeros(N, dtype=np.int64)
    onpath_out = np.zeros((N, W), dtype=np.int64)
    onpath_len = np.zeros(N, dtype=np.int64)
    t_conv = np.full(N, -1, dtype=np.int64)
    active = rows_all.copy()
    slot2d = active[:, None] * n + agent_idx[None, :]                 # (A, n)
    slot = slot2d.ravel()
    state_slot = np.repeat(state[active], n)
    alpha_act = alpha_all[slot]          # 随 active 收缩而重新切片
    om_act = om_all[slot]
    ad_act = ad_all[slot]
    history_pi: list[np.ndarray] = []
    history_eps: list[np.ndarray] = []
    history_ma: list[np.ndarray] = []
    shock = cfgs[0].demand_shock
    shock = None if shock is None else np.asarray(shock, dtype=float)
    boltz = cfgs[0].exploration == "boltzmann"
    temp = max(cfgs[0].temperature, 1e-12)
    eps_floor = float(cfgs[0].eps_floor)
    record_every = int(cfgs[0].record_every)

    max_T = int(limit_row.max())
    t = 0
    while active.size > 0 and t < max_T:
        A = active.size
        ns = A * n
        if boltz:
            z = Q[slot, state_slot, :] / temp
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            act_slot = _sample(rng, p)
        else:
            eps_slot = np.repeat(np.maximum(eps_floor, np.exp(-beta_row[active] * t)), n)
            uv = rng.random(2 * ns)
            rowv = Q[slot, state_slot, :]
            act_slot = np.where(uv[:ns] < eps_slot,
                                (uv[ns:] * m).astype(np.int64), rowv.argmax(axis=1))
        act = act_slot.reshape(A, n)

        if shock is None:
            pi = pi_tab[act[:, 0] * m + act[:, 1]] if n == 2 else \
                pi_tab[_profile_index_np(act, m)]
            pi_mean = pi.sum(axis=1) * (1.0 / n)
        else:
            prices = grid[act]
            uu = np.exp(spec.kappa * (spec.quality[None, :] - prices))
            den = uu.sum(axis=1) + np.exp(
                spec.kappa * shock[rng.integers(0, len(shock), A)])
            qq = uu / den[:, None]
            pi = (prices - spec.cost[None, :]) * qq
            pi_mean = pi.sum(axis=1) * (1.0 / n)

        if k == 1:
            next_state = act @ w_next
        elif k > 1:
            hist[active] = np.concatenate([hist[active][:, 1:, :], act[:, None, :]], axis=1)
            next_state = _encode_hist(hist[active], m)
        else:
            next_state = state[active]

        state_out[active] = next_state
        old_slot_state = state_slot
        state_slot = np.repeat(next_state, n)
        tgt = Q[slot, state_slot, :].max(axis=1)
        cur = Q[slot, old_slot_state, act_slot]
        cur *= om_act
        cur += pi.ravel() * alpha_act
        tgt *= ad_act
        cur += tgt
        Q[slot, old_slot_state, act_slot] = cur
        pi_ring[active, ptr] = pi_mean
        st_ring[active, ptr] = next_state
        ptr = (ptr + 1) % W
        cnt = min(cnt + 1, W)

        if record_every and (t % record_every == 0 or t == max_T - 1):
            # 用整行长度记录 (已冻结的行填 NaN), 避免不规则数组
            buf = np.full(N, np.nan)
            buf[active] = pi_mean
            history_pi.append(buf)
            ebuf = np.full(n * N, np.nan)
            ebuf[slot] = np.repeat(np.maximum(
                eps_floor, np.exp(-beta_row[active] * t)), n)
            history_eps.append(ebuf)
            mbuf = np.full(N, np.nan)                       # 移动平均 (图 10 用)
            mbuf[active] = pi_ring[active][:, _ring_slice(ptr, cnt, W)].mean(axis=1)
            history_ma.append(mbuf)

        t += 1
        if fixed:
            if t >= max_T:
                break
            continue

        # 收敛判据 (每 check_every 期抽样检查一次 argmax 映射)
        if t % check_every == 0:
            ng = np.argmax(Q[slot, old_slot_state, :], axis=1).astype(np.int16)
            ch = ng != best[slot, old_slot_state]
            best[slot, old_slot_state] = ng
            stable[slot] = np.where(ch, 0, stable[slot] + check_every)
        done = (stable[slot].reshape(A, n).min(axis=1) >= pat_row[active]) | \
               (t >= limit_row[active])
        if done.any():
            fin = active[done]
            _finalize(fin, pi_ring, st_ring, ptr, cnt, W, pi_out, onpath_out,
                      onpath_len, t_conv, t, fixed)
            # 冻结会话的极限策略取自其最终 Q 矩阵的精确 argmax (最低价优先)
            fslot = (fin[:, None] * n + agent_idx[None, :]).ravel()
            best[fslot] = np.argmax(Q[fslot], axis=-1).astype(np.int16)
            keep = ~done
            active = active[keep]
            slot2d = active[:, None] * n + agent_idx[None, :]
            slot = slot2d.ravel()
            state_slot = state_slot.reshape(A, n)[keep].ravel()
            alpha_act = alpha_all[slot]
            om_act = om_all[slot]
            ad_act = ad_all[slot]

    if active.size:                      # 到达硬上限的会话同样记录
        _finalize(active, pi_ring, st_ring, ptr, cnt, W, pi_out, onpath_out,
                  onpath_len, t_conv, t, fixed)
        fslot = (active[:, None] * n + agent_idx[None, :]).ravel()
        best[fslot] = np.argmax(Q[fslot], axis=-1).astype(np.int16)

    if active.size:                      # 到达硬上限的会话同样记录
        _finalize(active, pi_ring, st_ring, ptr, cnt, W, pi_out, onpath_out,
                  onpath_len, t_conv, t, fixed)

    longrun_actions = best.reshape(N, n, M)[rows_all, :, state_out]
    longrun_prices = grid[longrun_actions]
    conv = t_conv >= 0
    if fixed:
        conv = np.ones(N, dtype=bool)
        t_conv[:] = max_T
    cycles = classify_cycles(best.reshape(N, n, M), onpath_out, onpath_len, conv, m, n)

    res = SessionResult(
        converged=conv, t_conv=t_conv, pi_final=pi_out,
        longrun_actions=longrun_actions, longrun_prices=longrun_prices,
        cycle_type=cycles, limit_actions=best.reshape(N, n, M),
        onpath_states=onpath_out, env=env,
        meta={"alpha": alpha_row, "beta": beta_row, "delta": delta, "m": m, "k": k,
              "n": n, "M": M, "grid": grid, "spec": spec, "sessions": R, "points": P,
              "cfgs": cfgs},
    )
    if history_pi:
        res.history = {"pi": np.array(history_pi).T, "eps": np.array(history_eps).T,
                       "pi_ma": np.array(history_ma).T, "window": W,
                       "every": record_every, "env": env, "alpha": alpha_row,
                       "beta": beta_row}
    if return_q:
        # 已冻结会话的 Q 行保持在其收敛时刻的状态, 未收敛的保持最终状态
        res.Q_final = Q.reshape(N, n, M, m).copy()
    return res


def run_chunk(cfg: SimConfig, n_sessions: int, seed: int = 0,
              env: Optional[dict] = None) -> SessionResult:
    """单点运行 (run_batch 的特例)。"""
    return run_batch([cfg], n_sessions, seed=seed, env=env)


def _finalize(rows, pi_ring, st_ring, ptr, cnt, W, pi_out, onpath_out, onpath_len,
              t_conv, t, fixed) -> None:
    """冻结会话: 从环形缓冲取最近 cnt 期求 pi-bar, 并保存末端在轨状态。"""
    idx = (np.arange(-cnt, 0) + ptr) % W
    pi_out[rows] = pi_ring[rows][:, idx].mean(axis=1)
    onpath_out[rows, :cnt] = st_ring[rows][:, idx]
    onpath_len[rows] = cnt
    if not fixed:
        t_conv[rows] = t


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _state_weights(n: int, m: int) -> np.ndarray:
    w = np.empty(n)
    for j in range(n):
        w[j] = m ** (n - 1 - j)
    return w.astype(np.int64)


def _ring_slice(ptr: int, filled: int, W: int) -> np.ndarray:
    """环形缓冲中最新的 `filled` 个槽位, 按时间顺序。"""
    return (np.arange(-filled, 0) + ptr) % W


def _encode_hist(hist: np.ndarray, m: int) -> np.ndarray:
    R_, k, n = hist.shape
    out = np.zeros(R_, dtype=np.int64)
    for j in range(k):
        out = out * m
        for i in range(n):
            out = out * m + hist[:, j, i]
    return out


def _profile_index_np(actions: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros(actions.shape[0], dtype=np.int64)
    for j in range(actions.shape[1]):
        out = out * m + actions[:, j]
    return out


def _sample(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    c = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])[:, None]
    return np.clip((u > c).sum(axis=1), 0, probs.shape[1] - 1)


def classify_cycles(limit_actions: np.ndarray, onpath: np.ndarray,
                    onpath_len: np.ndarray, converged: np.ndarray,
                    m: int, n: int = 2) -> np.ndarray:
    """区分 恒定同价 / 恒定异价 / 周期 2 / 周期 >= 3 / 未收敛。

    从末端在轨状态出发, 沿"双方极限策略"映射迭代直到状态重复。
    """
    N, nn, M_states = limit_actions.shape
    out = np.full(N, CYCLE_NOTCONV, dtype=np.int8)
    if M_states <= 1:
        # k=0 (无记忆): 只有一个状态, 不存在价格周期, 直接归为"恒定"
        out[converged] = CYCLE_CONST_ASYM
        return out
    for r in range(N):
        if not converged[r] or onpath_len[r] == 0:
            continue
        s = int(onpath[r, -1])
        seen: dict[int, int] = {}
        seq: list[int] = []
        cap = nn * limit_actions.shape[2] + 2
        while s not in seen and len(seq) <= cap:
            seen[s] = len(seq)
            seq.append(s)
            a = limit_actions[r, :, s].astype(np.int64)
            s = int(a @ _state_weights(limit_actions.shape[1], m))
        period = len(seq) - seen.get(s, len(seq))
        if period <= 1:
            acts = limit_actions[r, :, int(seq[0])]
            out[r] = CYCLE_CONST_SYM if n > 1 and acts[0] == acts[1] else CYCLE_CONST_ASYM
        elif period == 2:
            out[r] = CYCLE_2
        else:
            out[r] = CYCLE_GE3
    return out


def merge(results: Sequence[SessionResult]) -> SessionResult:
    """沿会话维度拼批 (要求同一批内 alpha/beta 也相同)。"""
    cat = lambda key: np.concatenate([getattr(r, key) for r in results], axis=0)  # noqa: E731
    first = results[0]
    return SessionResult(
        converged=cat("converged"), t_conv=cat("t_conv"), pi_final=cat("pi_final"),
        longrun_actions=cat("longrun_actions"), longrun_prices=cat("longrun_prices"),
        cycle_type=cat("cycle_type"), limit_actions=cat("limit_actions"),
        onpath_states=cat("onpath_states"), env=first.env, meta=first.meta,
        Q_final=(None if first.Q_final is None else np.concatenate(
            [r.Q_final for r in results], axis=0)))