"""定位热点 + 检查 epsilon 轨迹。"""
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec                        # noqa: E402
from engine import SimConfig, build_env, run_batch  # noqa: E402


def main() -> None:
    periods = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    sessions = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000

    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    fixed_periods=periods, record_every=every)
    env = build_env(cfg)

    # --- eps 轨迹检查 (与 engine 内部公式一致) ---
    beta = np.full(sessions, 4e-6)
    act = np.arange(sessions)
    print("eps formula check:")
    for t in (0, 2000, 10000, periods):
        e = np.maximum(0.0, np.exp(-beta[act] * t))
        print(f"   t={t:>8,}  eps={e[0]:.6f}")

    pr = cProfile.Profile()
    pr.enable()
    res = run_batch([cfg], sessions, seed=3, env=env)
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("tottime").print_stats(12)

    h = res.history
    d = (h["pi"] - env["pi_nash"]) / (env["pi_mono"] - env["pi_nash"])
    print("\nrecord  t        eps_mean   meanDelta")
    for j in range(d.shape[1]):
        print(f"{j:>6d}  {j*every:>8,}  {float(np.mean(h['eps'][:, j])):.5f}  "
              f"{float(np.mean(d[:, j])):.4f}")


if __name__ == "__main__":
    main()