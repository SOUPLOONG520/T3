"""run_timescale.py -- 论文第 VI 节时间尺度（图 10 学习曲线 + 图 11 重新配对）入口。

--only fig10           只跑图 10
--only fig11           只跑图 11（自包含的 representative_small.npz 训练）
--only all             全部（默认）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "src"))

from timescale import run_learning_curve, run_rematch, save_json  # noqa: E402

RESULTS = os.path.join(THIS_DIR, "results")
FIGURES = os.path.join(THIS_DIR, "figures")


def _do_fig10(args) -> None:
    t0 = time.time()
    out = run_learning_curve(
        n_sessions=args.sessions,
        fixed_periods=args.fixed_periods,
        record_every=args.record_every,
        n_workers=1,
    )
    save_json(out, os.path.join(RESULTS, "timescale_fig10.json"))
    _plot_fig10(out)
    print(f"[fig10] {args.sessions} sessions × {args.fixed_periods} periods "
          f"in {time.time()-t0:.1f}s")


def _do_fig11(args) -> None:
    t0 = time.time()
    out = run_rematch(
        n_pairs=args.sessions,
        max_periods=args.max_periods,
        record_every=args.record_every,
        n_workers=1,
    )
    save_json(out, os.path.join(RESULTS, "timescale_fig11.json"))
    _plot_fig11(out)
    print(f"[fig11] {args.sessions} pairs × {args.max_periods} periods "
          f"in {time.time()-t0:.1f}s")


def _plot_fig10(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(out["time_axis"], dtype=float)
    d_mean = np.asarray(out["delta_mean"], dtype=float)
    d_std = np.asarray(out["delta_std"], dtype=float)
    d_bench = np.asarray(out["delta_bench"], dtype=float)
    # 截断到 d_mean 长度
    n_use = min(len(t), len(d_mean), len(d_std))
    t = t[:n_use]; d_mean = d_mean[:n_use]; d_std = d_std[:n_use]
    n_b = min(len(d_bench), n_use)
    d_bench = d_bench[:n_b]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t[:n_b], d_bench, ls="--", color="grey", lw=1.2,
            label="Exogenous-exploration benchmark")
    ax.plot(t, d_mean, color="C0", lw=1.2,
            label="Profit gain (avg over sessions)")
    ax.fill_between(t, d_mean - d_std, d_mean + d_std, color="C0", alpha=0.2)
    ax.set_xlabel("Time (periods)")
    ax.set_ylabel("Profit gain Δ")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, t[-1])
    ax.set_title("Figure 10: Average profit gain in representative experiment\n"
                "(moving average over last 100 repetitions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig10_learning_curve.png"), dpi=150)
    plt.close(fig)


def _plot_fig11(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(out["time_axis"], dtype=float)
    d_mean = np.asarray(out["delta_mean"], dtype=float)
    d_std = np.asarray(out["delta_std"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, d_mean, color="C1", lw=1.2, label="Profit gain (avg over pairs)")
    ax.fill_between(t, d_mean - d_std, d_mean + d_std, color="C1", alpha=0.2)
    ax.set_xlabel("Time (periods)")
    ax.set_ylabel("Profit gain Δ")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, t[-1])
    ax.set_title("Figure 11: Average profit gain for rematched pairs\n"
                "(moving average over last 100 repetitions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig11_rematch.png"), dpi=150)
    plt.close(fig)


import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        help="comma-separated subset of {fig10,fig11,all}")
    parser.add_argument("--sessions", type=int, default=50,
                        help="图 10 会话数 / 图 11 配对数")
    parser.add_argument("--fixed_periods", type=int, default=1_000_000)
    parser.add_argument("--record_every", type=int, default=5000)
    parser.add_argument("--max_periods", type=int, default=40_000,
                        help="图 11 续跑周期数")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    targets = (["fig10", "fig11"] if args.only == "all"
                else [t.strip() for t in args.only.split(",") if t.strip()])

    if "fig10" in targets:
        print("\n=== Running fig10 ===")
        _do_fig10(args)
    if "fig11" in targets:
        print("\n=== Running fig11 ===")
        _do_fig11(args)


if __name__ == "__main__":
    main()