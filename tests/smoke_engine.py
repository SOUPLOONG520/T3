"""引擎冒烟测试: 检查 Q-learning 是否学出超竞争价格, 并测速。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec                      # noqa: E402
from engine import SimConfig, build_env, run_chunk, CYCLE_NAMES  # noqa: E402


def main() -> None:
    periods = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    patience=100_000, max_periods=periods)
    env = build_env(cfg)
    print("grid      :", np.round(env["grid"], 4))
    print("p^N, p^M  :", round(env["p_nash"], 5), round(env["p_mono"], 5))
    print("pi^N, pi^M:", round(env["pi_nash"], 5), round(env["pi_mono"], 5))

    t0 = time.perf_counter()
    res = run_chunk(cfg, sessions, seed=1, env=env)
    dt = time.perf_counter() - t0
    print(f"\nran {periods} periods x {sessions} sessions in {dt:.1f}s "
          f"({periods / dt:,.0f} periods/s)")

    d = res.delta()
    print("\nfinal Delta per session: mean %.3f  min %.3f  max %.3f"
          % (np.nanmean(d), np.nanmin(d), np.nanmin(d) if False else np.nanmax(d)))
    print("longrun prices (firm1) unique:", np.unique(np.round(res.longrun_prices[:, 0], 4))[:8])
    print("converged sessions:", int(res.converged.sum()), "/", sessions)
    for code, name in CYCLE_NAMES.items():
        cnt = int((res.cycle_type == code).sum())
        if cnt:
            print(f"  cycle {name}: {cnt}")
    print("example limit strategy firm1 (state -> action) first 12 states:",
          res.limit_actions[0, 0, :12])


if __name__ == "__main__":
    main()