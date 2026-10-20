"""deviation.py -- Section IV.B: 偏离与惩罚 (Impulse response, Tables 2/3, Figures 4/5/6).

口径:
    - 从每个会话的长期状态出发 (1-cycle: 一个吸收态; 周期>1: 环上的每个节点各做一次偏离,
      算一个观测);
    - τ=0 双方按极限策略; τ=1 强制偏离方改用一个指定偏离价 (离散动作序号), 非偏离方仍按
      极限策略; τ≥2 双方都回到极限策略;
    - "事前价格" = 偏离方在 τ=0 的价格 (即吸收态的动作价);
    - Panel A: 非偏离方 τ=2 价格相对事前价格的相对变化;
    - Panel B: 偏离方 τ=2 价格相对 τ=1 价格的相对变化;
    - 偏离贴现利润变化: 用 analysis.deviation_gain, 路径从 τ=0 开始, δ=0.95.

依赖: src/analysis.py 中的 best_response_to_price, deviation_gain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core import encode_profile, profile_index
from analysis import best_response_to_price, deviation_gain


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _state_weights(n: int, m: int) -> np.ndarray:
    w = np.empty(n, dtype=np.int64)
    for j in range(n):
        w[j] = m ** (n - 1 - j)
    return w


def cycle_states_from_longrun(limit_row: np.ndarray, longrun_actions_row: np.ndarray,
                              m: int, n: int = 2, cap: int = 400) -> list[int]:
    """从 longrun_actions_row 对应的状态出发, 沿极限策略迭代得到环上节点列表。

    若环为单个吸收节点 (1-cycle), 返回 [s0]; 若为周期 L, 返回 L 个状态按访问顺序。
    """
    w = _state_weights(n, m)
    s = int(profile_index(longrun_actions_row, m))
    cycle = [s]
    for _ in range(cap):
        a = limit_row[:, s].astype(np.int64)
        s_next = int((a @ w).astype(np.int64))
        if s_next == s:
            return cycle
        if s_next in cycle:
            return cycle
        cycle.append(s_next)
        s = s_next
    return cycle


def absorbing_state(limit_row: np.ndarray, longrun_actions_row: np.ndarray,
                    m: int, n: int = 2) -> int:
    """找到 longrun_actions_row 对应的吸收态 (1-cycle 唯一; 周期 L: 返回任一环上节点).

    对周期 >1, 这里返回 engine 结束时的 state_out (即 limit_actions[r, :, state_out] =
    longrun_actions[r]). 周期内"事前价格"= cycle 内每个节点的 pre-shock 价.
    """
    w = _state_weights(n, m)
    s = int(profile_index(longrun_actions_row, m))
    return s


def impulse_response(limit_actions: np.ndarray, env: dict, deviator: int,
                     deviation_action: int, starting_state: int,
                     horizon: int = 15, n: int = 2) -> dict:
    """从 starting_state 出发, 做一次 impulse-response.

    τ=0 双方按 limit_actions; τ=1 偏离方改用 deviation_action; τ≥2 双方都按 limit_actions.

    Returns dict: prices_dev, prices_nondev, actions_dev, actions_nondev,
    profits_dev, profits_nondev, states.
    """
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    nondev = 1 - deviator
    w = _state_weights(n, m)

    states = np.zeros(horizon + 2, dtype=np.int64)
    actions = np.zeros((horizon + 1, n), dtype=np.int64)
    states[0] = int(starting_state)
    for t in range(horizon + 1):
        actions[t] = limit_actions[:, states[t]]
        if t == 1:
            actions[t, deviator] = int(deviation_action)
        states[t + 1] = int((actions[t] @ w).astype(np.int64))

    a_dev = actions[:, deviator]
    a_nondev = actions[:, nondev]
    prices_dev = grid[a_dev]
    prices_nondev = grid[a_nondev]
    prof = pi_tab[encode_profile(actions.astype(np.int64), m)]

    return {
        'prices_dev': prices_dev,
        'prices_nondev': prices_nondev,
        'profits_dev': prof[:, deviator],
        'profits_nondev': prof[:, nondev],
        'actions_dev': a_dev,
        'actions_nondev': a_nondev,
        'states': states,
    }


def baseline_path(limit_actions: np.ndarray, env: dict, starting_state: int,
                  horizon: int = 15, n: int = 2) -> np.ndarray:
    """双方都按极限策略的基线逐期利润, 返回 (horizon+1, n) 数组."""
    pi_tab = env['pi_table']
    grid = env['grid']
    m = len(grid)
    w = _state_weights(n, m)
    actions = np.zeros((horizon + 1, n), dtype=np.int64)
    states = np.zeros(horizon + 2, dtype=np.int64)
    states[0] = int(starting_state)
    for t in range(horizon + 1):
        actions[t] = limit_actions[:, states[t]]
        states[t + 1] = int((actions[t] @ w).astype(np.int64))
    return pi_tab[encode_profile(actions.astype(np.int64), m)]


# --------------------------------------------------------------------------- #
#  Data collection for Tables 2 & 3
# --------------------------------------------------------------------------- #
@dataclass
class DeviationTable:
    """Tables 2 & 3 的聚合结果 (按事前价格 + 偏离价交叉)."""
    preshock_indices: list[int]
    preshock_prices: list[float]
    deviation_indices: list[int]
    deviation_prices: list[float]
    counts: np.ndarray                 # (P, D) 每格观测数
    panel_a: np.ndarray               # (P, D) 非偏离方 τ=2 vs preshock 相对变化
    panel_b: np.ndarray               # (P, D) 偏离方 τ=2 vs τ=1 相对变化
    panel_c: np.ndarray               # (P, D) 平均偏离贴现利润变化 (Table 3 Panel A)
    panel_d: np.ndarray               # (P, D) 无利可图偏离占比 (Table 3 Panel B)
    frequency: np.ndarray             # (P,)   每行事前价格的观测频率

    def to_csv(self, path: str, every: int = 1) -> None:
        """写出 Table 2 & 3 风格 CSV; every=1 全量, every=2 论文风格 (每隔 1)."""
        if every == 1:
            keep_d = list(range(len(self.deviation_indices)))
        else:
            keep_d = list(range(0, len(self.deviation_indices), every))
        dev_p = [round(self.deviation_prices[i], 4) for i in keep_d]
        preshock_p = [round(p, 4) for p in self.preshock_prices]

        def fmt(x):
            if not np.isfinite(x):
                return ''
            return f"{x:.3f}"

        def cell(v):
            return str(v)

        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Preshock price, Frequency")
            for p in dev_p:
                f.write(f", {p}")
            f.write('\n')

            f.write("Panel A. Relative price change by the nondeviating agent in period tau=2\n")
            for i, pp in enumerate(preshock_p):
                row = [cell(pp), f"{self.frequency[i]:.4f}"]
                for j in keep_d:
                    row.append(fmt(self.panel_a[i, j]))
                f.write(', '.join(row) + '\n')

            f.write("\nPanel B. Relative price change by the deviating agent in period tau=2 wrt tau=1\n")
            for i, pp in enumerate(preshock_p):
                row = [cell(pp), f"{self.frequency[i]:.4f}"]
                for j in keep_d:
                    row.append(fmt(self.panel_b[i, j]))
                f.write(', '.join(row) + '\n')

            f.write("\nTable 3 Panel A. Average percentage gain from the deviation (discounted)\n")
            for i, pp in enumerate(preshock_p):
                row = [cell(pp), f"{self.frequency[i]:.4f}"]
                for j in keep_d:
                    row.append(fmt(self.panel_c[i, j]))
                f.write(', '.join(row) + '\n')

            f.write("\nTable 3 Panel B. Frequency of unprofitable deviations\n")
            for i, pp in enumerate(preshock_p):
                row = [cell(pp), f"{self.frequency[i]:.4f}"]
                for j in keep_d:
                    row.append(fmt(self.panel_d[i, j]))
                f.write(', '.join(row) + '\n')


def build_deviation_tables(
    res,
    env: dict,
    delta: float,
    deviator: int = 0,
    horizon: int = 15,
    n: int = 2,
    max_deviation_idx: int | None = None,
    preshock_min_idx: int = 5,
) -> DeviationTable:
    """遍历所有 (session, cycle-state) 观测, 计算 Tables 2/3 的面板数据.

    每个观测:
      - pre-shock = grid[limit_actions[deviator, s_init]]
      - 非偏离方事前动作 = limit_actions[nondev, s_init]
      - 对每个 deviation_idx < pre-shock: 做 impulse_response, 计算 Panel A/B/C/D.

    Returns: DeviationTable
    """
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    nondev = 1 - deviator
    w = _state_weights(n, m)

    if max_deviation_idx is None:
        max_deviation_idx = m - 1 - 1  # grid[m-1] is highest, so max_dev = m-2

    preshock_indices = list(range(preshock_min_idx, m))
    deviation_indices = list(range(0, max_deviation_idx + 1))
    P = len(preshock_indices)
    D = len(deviation_indices)
    preshock_prices = [float(grid[i]) for i in preshock_indices]
    deviation_prices = [float(grid[i]) for i in deviation_indices]

    sum_A = np.zeros((P, D)); sum_B = np.zeros((P, D))
    sum_C = np.zeros((P, D)); neg_C = np.zeros((P, D), dtype=np.int64)
    counts = np.zeros((P, D), dtype=np.int64)
    freq_counts = np.zeros(P, dtype=np.int64)

    N = int(res.converged.shape[0])
    for r in range(N):
        if not res.converged[r]:
            continue
        limit_row = res.limit_actions[r]
        longrun = res.longrun_actions[r]
        cycle = cycle_states_from_longrun(limit_row, longrun, m, n=n)
        for s_init in cycle:
            preshock_idx = int(limit_row[deviator, s_init])
            if preshock_idx not in preshock_indices:
                continue
            p_idx = preshock_indices.index(preshock_idx)
            freq_counts[p_idx] += 1
            preshock_price = float(grid[preshock_idx])

            # 基线路径 (τ=0..horizon, 双方都按极限策略)
            base = baseline_path(limit_row, env, s_init, horizon=horizon, n=n)

            for j, dev_idx in enumerate(deviation_indices):
                if dev_idx >= preshock_idx:
                    continue  # 必须是价格下调; 对角与上方格留空
                ir = impulse_response(limit_row, env, deviator, dev_idx, s_init,
                                      horizon=horizon, n=n)
                # Panel A: 非偏离方 τ=2 vs pre-shock
                a_rel = (ir['prices_nondev'][2] - preshock_price) / preshock_price
                # Panel B: 偏离方 τ=2 vs τ=1
                b_rel = (ir['prices_dev'][2] - ir['prices_dev'][1]) / ir['prices_dev'][1]
                # Panel C: 偏离贴现利润变化
                c_val = deviation_gain(ir['profits_dev'], base[:, deviator], delta)
                sum_A[p_idx, j] += a_rel
                sum_B[p_idx, j] += b_rel
                sum_C[p_idx, j] += c_val
                neg_C[p_idx, j] += int(c_val <= 0)
                counts[p_idx, j] += 1

    safe = np.maximum(counts, 1)
    panel_a = sum_A / safe
    panel_b = sum_B / safe
    panel_c = sum_C / safe
    panel_d = neg_C / safe
    total_obs = int(freq_counts.sum())
    frequency = freq_counts / max(total_obs, 1)

    # 把对角与上方格 (deviation >= preshock) 标 NaN
    for i, ps_idx in enumerate(preshock_indices):
        for j, dv_idx in enumerate(deviation_indices):
            if dv_idx >= ps_idx:
                panel_a[i, j] = np.nan
                panel_b[i, j] = np.nan
                panel_c[i, j] = np.nan
                panel_d[i, j] = np.nan

    return DeviationTable(
        preshock_indices=preshock_indices,
        preshock_prices=preshock_prices,
        deviation_indices=deviation_indices,
        deviation_prices=deviation_prices,
        counts=counts,
        panel_a=panel_a,
        panel_b=panel_b,
        panel_c=panel_c,
        panel_d=panel_d,
        frequency=frequency,
    )


# --------------------------------------------------------------------------- #
#  Figures 4 / 5 / 6: 偏离脉冲响应
# --------------------------------------------------------------------------- #
def figure4_path(res, env: dict, deviator: int = 0, horizon: int = 15,
                 n: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                      float, float, float]:
    """Figure 4 平均脉冲响应: 偏离 = 对非偏离方事前价格的一阶最优回应.

    Returns: mean_prices_dev, mean_prices_nondev, mean_long_dev, mean_long_nondev,
             p_nash, p_mono, mean_longrun_price
    """
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    nondev = 1 - deviator

    all_dev = []
    all_nondev = []
    long_dev = []
    long_nondev = []
    n_obs = 0

    N = int(res.converged.shape[0])
    for r in range(N):
        if not res.converged[r]:
            continue
        limit_row = res.limit_actions[r]
        longrun = res.longrun_actions[r]
        cycle = cycle_states_from_longrun(limit_row, longrun, m, n=n)
        for s_init in cycle:
            nondev_act = int(limit_row[nondev, s_init])
            dev_action = best_response_to_price(pi_tab, m, n, deviator, nondev_act)
            ir = impulse_response(limit_row, env, deviator, dev_action, s_init,
                                  horizon=horizon, n=n)
            all_dev.append(ir['prices_dev'])
            all_nondev.append(ir['prices_nondev'])
            long_dev.append(float(grid[limit_row[deviator, s_init]]))
            long_nondev.append(float(grid[limit_row[nondev, s_init]]))
            n_obs += 1

    all_dev = np.asarray(all_dev)
    all_nondev = np.asarray(all_nondev)
    mean_dev = all_dev.mean(axis=0)
    mean_nondev = all_nondev.mean(axis=0)
    mean_longrun_dev = float(np.mean(long_dev))
    mean_longrun_nondev = float(np.mean(long_nondev))

    return mean_dev, mean_nondev, mean_longrun_dev, mean_longrun_nondev, \
        float(env['p_nash']), float(env['p_mono']), (mean_longrun_dev + mean_longrun_nondev) / 2


def figure5_distributions(res, env: dict, deviator: int = 0, horizon: int = 15,
                           n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Figure 5 分布图 (仅 1-cycle 会话): 返回 (delta_dev, delta_nondev),
    各为 (n_obs, horizon+1) 数组, 值 = 当期价格 - 长期价格 (按各自企业的长期价格)."""
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    nondev = 1 - deviator

    all_dev = []
    all_nondev = []
    long_dev = []
    long_nondev = []

    N = int(res.converged.shape[0])
    for r in range(N):
        if not res.converged[r] or res.cycle_type[r] not in (0, 1):
            continue
        limit_row = res.limit_actions[r]
        longrun = res.longrun_actions[r]
        s_init = int(profile_index(longrun, m))  # 1-cycle 吸收态
        nondev_act = int(limit_row[nondev, s_init])
        dev_action = best_response_to_price(pi_tab, m, n, deviator, nondev_act)
        ir = impulse_response(limit_row, env, deviator, dev_action, s_init,
                              horizon=horizon, n=n)
        all_dev.append(ir['prices_dev'])
        all_nondev.append(ir['prices_nondev'])
        long_dev.append(float(grid[limit_row[deviator, s_init]]))
        long_nondev.append(float(grid[limit_row[nondev, s_init]]))

    all_dev = np.asarray(all_dev)
    all_nondev = np.asarray(all_nondev)
    long_dev = np.asarray(long_dev)[:, None]
    long_nondev = np.asarray(long_nondev)[:, None]
    return all_dev - long_dev, all_nondev - long_nondev


def figure6_path(res, env: dict, deviator: int = 0, horizon: int = 15,
                 n: int = 2, drop: int = 2) -> tuple:
    """Figure 6 较小偏离 price, 选能体现 overshoot 的偏离价.

    选择规则: 每个观测取 deviation_idx = max(preshock_idx - drop, 0),
    保证偏离价明显小于事前价 (论文约定 "smaller price cut"). 若该 preshock_idx - drop
    仍 >= preshock_idx (极小 pre-shock), 则取 preshock_idx - 1.

    Returns: (mean_dev, mean_nondev, mean_longrun_dev, mean_longrun_nondev,
              p_nash, p_mono, mean_longrun, chosen_drop)
    """
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    nondev = 1 - deviator

    all_dev = []
    all_nondev = []
    long_dev = []
    long_nondev = []
    n_obs = 0

    N = int(res.converged.shape[0])
    for r in range(N):
        if not res.converged[r]:
            continue
        limit_row = res.limit_actions[r]
        longrun = res.longrun_actions[r]
        cycle = cycle_states_from_longrun(limit_row, longrun, m, n=n)
        for s_init in cycle:
            preshock_idx = int(limit_row[deviator, s_init])
            dev_idx = max(preshock_idx - drop, 0)
            if dev_idx >= preshock_idx:
                dev_idx = preshock_idx - 1 if preshock_idx > 0 else 0
            ir = impulse_response(limit_row, env, deviator, dev_idx, s_init,
                                  horizon=horizon, n=n)
            all_dev.append(ir['prices_dev'])
            all_nondev.append(ir['prices_nondev'])
            long_dev.append(float(grid[limit_row[deviator, s_init]]))
            long_nondev.append(float(grid[limit_row[nondev, s_init]]))
            n_obs += 1

    all_dev = np.asarray(all_dev)
    all_nondev = np.asarray(all_nondev)
    mean_dev = all_dev.mean(axis=0)
    mean_nondev = all_nondev.mean(axis=0)
    mean_longrun_dev = float(np.mean(long_dev))
    mean_longrun_nondev = float(np.mean(long_nondev))

    return (mean_dev, mean_nondev, mean_longrun_dev, mean_longrun_nondev,
            float(env['p_nash']), float(env['p_mono']),
            (mean_longrun_dev + mean_longrun_nondev) / 2, drop)


# --------------------------------------------------------------------------- #
#  Summary stats
# --------------------------------------------------------------------------- #
def deviation_summary_stats(table: DeviationTable) -> dict:
    """汇总无利可图占比、利润变化、惩罚强度、事前价 1.78 细值."""
    finite_d = np.isfinite(table.panel_d)
    unprof_frac = float(table.panel_d[finite_d].mean()) if finite_d.any() else float('nan')
    finite_c = np.isfinite(table.panel_c)
    mean_gain = float(table.panel_c[finite_c].mean()) if finite_c.any() else float('nan')
    finite_a = np.isfinite(table.panel_a)
    punishment = float(table.panel_a[finite_a].mean()) if finite_a.any() else float('nan')

    target_preshock = 1.78
    best_idx = int(np.argmin(np.abs(np.array(table.preshock_prices) - target_preshock)))
    target_price_actual = float(table.preshock_prices[best_idx])
    panel_a_at = table.panel_a[best_idx, :].copy()
    panel_b_at = table.panel_b[best_idx, :].copy()
    panel_c_at = table.panel_c[best_idx, :].copy()
    panel_d_at = table.panel_d[best_idx, :].copy()

    return {
        'paper_targets': {
            'unprofitable_fraction_min': 0.95,
            'mean_discounted_gain_range': (-0.04, -0.03),
            'panel_a_at_preshock_178': -0.13,
        },
        'unprofitable_fraction': unprof_frac,
        'mean_discounted_gain': mean_gain,
        'mean_punishment_nondev_tau2': punishment,
        'discount_window': 'tau=0..15 (16 periods, horizon=15); delta=0.95; '
                           'paths start at tau=0 with both firms playing the limit strategy.',
        'target_preshock_price': target_price_actual,
        'target_preshock_index': int(table.preshock_indices[best_idx]),
        'panel_a_at_target': {  # 论文 Figure 4 注释: 1.78 行平均约 -13%
            'panel_a_mean': float(np.nanmean(panel_a_at)) if np.isfinite(panel_a_at).any() else float('nan'),
            'panel_a_per_deviation': [float(v) if np.isfinite(v) else None for v in panel_a_at],
            'panel_b_mean': float(np.nanmean(panel_b_at)) if np.isfinite(panel_b_at).any() else float('nan'),
            'panel_c_mean': float(np.nanmean(panel_c_at)) if np.isfinite(panel_c_at).any() else float('nan'),
            'panel_d_mean': float(np.nanmean(panel_d_at)) if np.isfinite(panel_d_at).any() else float('nan'),
        },
        'deviation_prices': [float(p) for p in table.deviation_prices],
        'frequency_at_target': float(table.frequency[best_idx]),
        'observation_counts': {
            'total': int(np.nansum(table.counts)),
            'per_preshock': [int(c) for c in table.counts.sum(axis=1)],
        },
    }