"""诊断: 学习轨迹 + 收敛时刻 + 速度 (大 batch)。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec                       # noqa: E402
from engine import SimConfig, build_env, run_batch, CYCLE_NAMES  # noqa: E402


def main() -> None:
    periods = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000

    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    fixed_periods=periods, record_every=every, profit_window=1000)
    env = build_env(cfg)
    t0 = time.perf_counter()
    res = run_batch([cfg], sessions, seed=7, env=env)
    dt = time.perf_counter() - t0
    print(f"{periods:,} periods x {sessions} sessions in {dt:.1f}s "
          f"({periods/dt:,.0f} periods/s, {periods*sessions/dt:,.0f} row-periods/s)")

    h = res.history
    d = (h["pi"] - env["pi_nash"]) / (env["pi_mono"] - env["pi_nash"])
    dmean = np.nanmean(d, axis=0)
    print("\n  t        eps      mean Delta   mean price(uniform-random proxy)")
    for j in range(len(dmean)):
        t = j * every
        eps = float(np.mean(h["eps"][j]))
        print(f"{t:>9,}  {eps:8.5f}  {dmean[j]:>9.3f}")
    print("\np^N = %.4f  p^M = %.4f  grid = %s"
          % (env["p_nash"], env["p_mono"], np.round(env["grid"], 3)))
    print("longrun price distribution (firm 1):")
    u, c = np.unique(np.round(res.longrun_prices[:, 0], 4), return_counts=True)
    for uu, cc in zip(u, c):
        print("   %.4f  %5.1f%%" % (uu, 100 * cc / sessions))
    for code, name in CYCLE_NAMES.items():
        cnt = int((res.cycle_type == code).sum())
        if cnt:
            print(f"  cycle {name}: {cnt} ({100*cnt/sessions:.1f}%)")


if __name__ == "__main__":
    main()