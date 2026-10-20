"""Sanity checks for core.py against numbers printed in the paper."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec, benchmark_prices, price_grid, payoff_table, init_q_values  # noqa: E402


def main() -> None:
    spec = MarketSpec(n=2, quality=(2.0, 2.0), cost=(1.0, 1.0), a0=0.0, mu=0.25)
    bm = benchmark_prices(spec)
    print("kappa (exponent coefficient) =", spec.kappa)
    print("p^N  =", np.round(bm["p_nash"], 5), " margin =", round(bm["margin_nash"], 4))
    print("p^M  =", np.round(bm["p_mono"], 5), " margin =", round(bm["margin_mono"], 4))
    print("pi^N =", np.round(bm["pi_nash"], 5), " pi^M =", np.round(bm["pi_mono"], 5))
    print("q^N  =", np.round(bm["q_nash"], 5))

    grid = price_grid(bm["p_nash"][0], bm["p_mono"][0], m=15, xi=0.1)
    print("\nprice grid (15 points):")
    print(np.round(grid, 4))
    r2 = np.round(grid, 2)
    print("rounded to 2dp:", list(r2))
    tab2_dev = [1.43, 1.51, 1.58, 1.66, 1.74, 1.82, 1.89]
    tab2_pre = [1.62, 1.66, 1.70, 1.74, 1.78, 1.82, 1.85, 1.89, 1.93, 1.97]
    print("Table 2 deviation prices present in grid:",
          all(any(abs(g - d) < 0.005 for g in grid) for d in tab2_dev))
    print("Table 2 preshock prices present in grid:",
          all(any(abs(g - d) < 0.005 for g in grid) for d in tab2_pre))

    # delta at grid points around p^N  (paper: ~-2% if Nash approximated by defect,
    # ~12% if approximated by excess)
    pi_tab = payoff_table(spec, grid)
    pi_n, pi_m = bm["pi_nash_avg"], bm["pi_mono_avg"]
    print("\npi^N = %.5f   pi^M = %.5f   gap = %.5f" % (pi_n, pi_m, pi_m - pi_n))
    for a in range(6):
        idx = a * 15 + a            # symmetric profile (a, a)
        pi = pi_tab[idx, 0]
        print("  symmetric price %.4f -> pi = %.5f -> Delta = %+.4f"
              % (grid[a], pi, (pi - pi_n) / (pi_m - pi_n)))

    q0 = init_q_values(spec, pi_tab, 15, 0.95)
    print("\nQ_0 (discounted random-rival payoff), firm 1:", np.round(q0[0], 4))
    print("state/action cells for n=2, m=15, k=1:", 15 ** 2 * 15)


if __name__ == "__main__":
    main()