"""朴素参考实现 (逐会话 Python 循环) vs 向量化引擎的交叉验证。

参考实现严格按论文式 (4)/(5)/(6)/(7)/(8) 逐条写, 不做任何批量优化;
两者在同一参数下运行, 比较逐期每企业平均利润的横截面均值轨迹。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec, payoff_table, profile_index          # noqa: E402
from engine import SimConfig, build_env, run_batch                 # noqa: E402


def naive_session(cfg: SimConfig, env: dict, periods: int,
                  rng: np.random.Generator) -> np.ndarray:
    """单个会话的朴素实现, 返回逐期每企业平均利润。"""
    grid, pi_tab = env["grid"], env["pi_table"]
    m, n, delta = cfg.m, cfg.spec.n, cfg.delta
    M = m ** (n * cfg.memory) if cfg.memory > 0 else 1
    q0 = env["q0"]
    Q = [np.tile(q0[i][None, :], (M, 1)).copy() for i in range(n)]
    pi_path = np.zeros(periods)
    state = int(rng.integers(0, M)) if cfg.memory > 0 else 0
    for t in range(periods):
        eps = max(cfg.eps_floor, float(np.exp(-cfg.beta * t)))
        acts = []
        for i in range(n):
            if rng.random() < eps:
                acts.append(int(rng.integers(0, m)))
            else:
                acts.append(int(np.argmax(Q[i][state])))     # 最低价优先
        pi = pi_tab[profile_index(acts, m)]
        pi_path[t] = pi.mean()
        next_state = profile_index(acts, m)
        for i in range(n):
            tgt = Q[i][next_state].max()
            Q[i][state, acts[i]] = ((1 - cfg.alpha) * Q[i][state, acts[i]]
                                    + cfg.alpha * (pi[i] + delta * tgt))
        state = next_state
    return pi_path


def main() -> None:
    periods = int(sys.argv[1]) if len(sys.argv) > 1 else 12_000
    sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    fixed_periods=periods, record_every=every, profit_window=every,
                    patience=10 ** 9)
    env = build_env(cfg)

    t0 = time.perf_counter()
    ref = np.array([naive_session(cfg, env, periods, np.random.default_rng(1000 + r))
                    for r in range(sessions)])
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = run_batch([cfg], sessions, seed=12345, env=env)
    t_vec = time.perf_counter() - t0

    K = 1                                     # 比较最后一个移动平均窗口
    ref_win = ref[:, -(K * every):].mean(axis=1)      # 每会话在末窗口的均值
    vec_win = res.history["pi_ma"][:, -K:].mean(axis=1)  # 向量化同样取移动平均

    print(f"naive {t_ref:6.1f}s   vectorized {t_vec:5.2f}s   speedup x{t_ref / max(t_vec, 1e-9):.0f}")
    print(f"\n末 {K * every} 期的每会话平均利润 (pi^N={env['pi_nash']:.4f}, "
          f"pi^M={env['pi_mono']:.4f})")
    print("  session      naive         vectorized")
    for r in range(sessions):
        print(f"     {r:2d}     {ref_win[r]:.5f}       {vec_win[r]:.5f}")
    print(f"\n mean  naive {ref_win.mean():.5f}   vectorized {vec_win.mean():.5f}   "
          f"diff {abs(ref_win.mean() - vec_win.mean()):.5f}")
    print(f"  sd   naive {ref_win.std():.5f}   vectorized {vec_win.std():.5f}")
    gap = abs(ref_win.mean() - vec_win.mean())
    tol = 3 * ref_win.std() / np.sqrt(sessions) + 1e-4
    print("\nVERDICT:", "OK" if gap < tol else "MISMATCH (超出抽样误差)")


if __name__ == "__main__":
    main()