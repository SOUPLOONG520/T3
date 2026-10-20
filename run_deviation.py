"""run_deviation.py -- 任务 A: Section IV.B 偏离与惩罚.

读取 representative.npz, 跑:
  - 图 4 (偏离到静态最优回应, 平均脉冲响应)
  - 图 5 (1-cycle 子样本, 偏离方/非偏离方的价格变化分布)
  - 图 6 (较小偏离, 体现 overshooting)
  - 表 2 Panel A/B (相对价格变化)
  - 表 3 Panel A/B (贴现利润变化 & 无利可图频率)
  - 摘要 JSON (含 1.78 行的细值)
  - 中英文 README

CLI: --input / --out-prefix / --deviator / --horizon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "src"))

from engine import SessionResult  # noqa: E402
from deviation import (  # noqa: E402
    build_deviation_tables,
    deviation_summary_stats,
    figure4_path,
    figure5_distributions,
    figure6_path,
)


def _to_session_result(d: np.lib.npyio.NpzFile) -> SessionResult:
    env = {
        "grid": d["grid"],
        "pi_table": d["pi_table"],
        "p_nash": float(d["p_nash"]),
        "p_mono": float(d["p_mono"]),
        "pi_nash": float(d["pi_nash"]),
        "pi_mono": float(d["pi_mono"]),
    }
    return SessionResult(
        converged=d["converged"],
        t_conv=d["t_conv"],
        pi_final=d["pi_final"],
        longrun_actions=d["longrun_actions"],
        longrun_prices=d["longrun_prices"],
        cycle_type=d["cycle_type"],
        limit_actions=d["limit_actions"],
        onpath_states=d["onpath_states"],
        env=env,
        meta={"delta": float(d["delta"])},
    )


def _set_axes_en(ax):
    ax.tick_params(axis="both", which="both", labelsize=10)


def _plot_figure4(mean_dev, mean_nondev, mean_longrun_dev, mean_longrun_nondev,
                  p_nash, p_mono, mean_longrun, out_path):
    tau = np.arange(len(mean_dev))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau, mean_dev, color="C0", lw=1.6, label="Deviating agent")
    ax.plot(tau, mean_nondev, color="C3", lw=1.6, label="Nondeviating agent")
    ax.axhline(p_nash, color="gray", lw=0.9, ls="--", label="Nash price")
    ax.axhline(p_mono, color="gray", lw=0.9, ls=":", label="Monopoly price")
    ax.axhline(mean_longrun, color="black", lw=0.9, ls="-", label="Long-run price")
    ax.set_xlabel(r"Time $\tau$")
    ax.set_ylabel("Price")
    ax.set_xlim(0, len(mean_dev) - 1)
    ax.set_ylim(1.45, 1.95)
    ax.set_title("Figure 4: Impulse response to one-period deviation (static best response)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _set_axes_en(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_figure5(delta_dev, delta_nondev, out_path):
    # delta_dev: (n_obs, horizon+1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    titles = ["Deviating agent",
               "Nondeviating agent"]
    for ax, data, title in zip(axes, (delta_dev, delta_nondev), titles):
        if data.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center")
            ax.set_title(title)
            continue
        tau = np.arange(data.shape[1])
        med = np.nanmean(data, axis=0)
        q25 = np.nanquantile(data, 0.25, axis=0)
        q75 = np.nanquantile(data, 0.75, axis=0)
        lo = np.nanmin(data, axis=0)
        hi = np.nanmax(data, axis=0)
        ax.fill_between(tau, q25, q75, color="C0", alpha=0.25, label="25-75 pct")
        ax.plot(tau, med, color="black", lw=1.4, label="mean")
        ax.plot(tau, lo, color="C0", lw=0.6, ls="--", label="range")
        ax.plot(tau, hi, color="C0", lw=0.6, ls="--")
        ax.axhline(0.0, color="black", lw=0.4)
        ax.set_xlabel(r"Time $\tau$")
        ax.set_title(title)
        ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
        _set_axes_en(ax)
    axes[0].set_ylabel(r"Price $-$ long-run price")
    fig.suptitle("Figure 5: Distribution of price changes (1-cycle sessions only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_figure6(mean_dev, mean_nondev, mean_longrun_dev, mean_longrun_nondev,
                  p_nash, p_mono, mean_longrun, drop, out_path):
    tau = np.arange(len(mean_dev))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau, mean_dev, color="C0", lw=1.6, label="Deviating agent")
    ax.plot(tau, mean_nondev, color="C3", lw=1.6, label="Nondeviating agent")
    ax.axhline(p_nash, color="gray", lw=0.9, ls="--", label="Nash price")
    ax.axhline(p_mono, color="gray", lw=0.9, ls=":", label="Monopoly price")
    ax.axhline(mean_longrun, color="black", lw=0.9, ls="-", label="Long-run price")
    ax.set_xlabel(r"Time $\tau$")
    ax.set_ylabel("Price")
    ax.set_xlim(0, len(mean_dev) - 1)
    ax.set_ylim(1.45, 1.95)
    ax.set_title(f"Figure 6: Overshooting impulse response (deviation = preshock - {drop} grid steps)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _set_axes_en(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(THIS_DIR, "results", "representative.npz"))
    parser.add_argument("--out-prefix", default=os.path.join(THIS_DIR, "results"))
    parser.add_argument("--fig-prefix", default=os.path.join(THIS_DIR, "figures"))
    parser.add_argument("--deviator", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--drop", type=int, default=2, help="Figure 6: preshock - drop grid steps")
    args = parser.parse_args(argv)

    os.makedirs(args.out_prefix, exist_ok=True)
    os.makedirs(args.fig_prefix, exist_ok=True)

    print(f"[deviation] loading {args.input}")
    d = np.load(args.input)
    res = _to_session_result(d)
    env = res.env
    delta = float(d["delta"])
    n_conv = int(res.converged.sum())
    print(f"[deviation] {res.converged.shape[0]} sessions, {n_conv} converged, "
          f"delta={delta}, p_nash={env['p_nash']:.4f}, p_mono={env['p_mono']:.4f}")

    t0 = time.time()

    # ---- Tables 2 & 3 (full grid) ------------------------------------------ #
    print("[deviation] building Tables 2/3 (full grid)")
    table = build_deviation_tables(res, env, delta, deviator=args.deviator,
                                   horizon=args.horizon)
    table.to_csv(os.path.join(args.out_prefix, "deviation_table2.csv"), every=2)
    table.to_csv(os.path.join(args.out_prefix, "deviation_table2_full.csv"), every=1)
    table.to_csv(os.path.join(args.out_prefix, "deviation_table3.csv"), every=1)

    # ---- Summary stats ----------------------------------------------------- #
    summary = deviation_summary_stats(table)
    summary["observation_counts"]["n_converged_sessions"] = n_conv
    summary["observation_counts"]["n_total_sessions"] = int(res.converged.shape[0])
    summary["n_converged_sessions"] = n_conv
    summary["n_total_sessions"] = int(res.converged.shape[0])
    summary["delta_value"] = float(np.mean((res.pi_final - env["pi_nash"]) /
                                   (env["pi_mono"] - env["pi_nash"])))
    summary["delta_mean"] = float(np.mean(
        (res.pi_final - env["pi_nash"]) / (env["pi_mono"] - env["pi_nash"])))
    summary["p_nash"] = float(env["p_nash"])
    summary["p_mono"] = float(env["p_mono"])
    summary["grid"] = [float(g) for g in env["grid"]]
    summary["deviator"] = args.deviator
    summary["horizon"] = args.horizon
    summary["drop_for_fig6"] = args.drop

    with open(os.path.join(args.out_prefix, "deviation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Figure 4 ------------------------------------------------------ #
    print("[deviation] computing Figure 4 (static best response)")
    (mean_dev4, mean_nondev4, mean_long4_dev, mean_long4_nondev,
     p_nash, p_mono, mean_long4) = figure4_path(
            res, env, deviator=args.deviator, horizon=args.horizon)
    _plot_figure4(mean_dev4, mean_nondev4, mean_long4_dev, mean_long4_nondev,
                  p_nash, p_mono, mean_long4,
                  os.path.join(args.fig_prefix, "fig4_punishment_impulse.png"))

    # ---- Figure 5 ------------------------------------------------------ #
    print("[deviation] computing Figure 5 (1-cycle distribution)")
    delta_dev5, delta_nondev5 = figure5_distributions(
        res, env, deviator=args.deviator, horizon=args.horizon)
    _plot_figure5(delta_dev5, delta_nondev5,
                  os.path.join(args.fig_prefix, "fig5_impulse_distribution.png"))

    # ---- Figure 6 ------------------------------------------------------ #
    print(f"[deviation] computing Figure 6 (overshooting, drop={args.drop})")
    (mean_dev6, mean_nondev6, mean_long6_dev, mean_long6_nondev,
     _, _, mean_long6, drop_used) = figure6_path(
            res, env, deviator=args.deviator, horizon=args.horizon, drop=args.drop)
    _plot_figure6(mean_dev6, mean_nondev6, mean_long6_dev, mean_long6_nondev,
                  p_nash, p_mono, mean_long6, drop_used,
                  os.path.join(args.fig_prefix, "fig6_overshooting.png"))

    print(f"[deviation] done in {time.time() - t0:.1f}s")
    print(f"[deviation] unprofitable fraction = {summary['unprofitable_fraction']:.3f} "
          f"(paper >0.95)")
    print(f"[deviation] mean discounted gain = {summary['mean_discounted_gain']:.4f} "
          f"(paper ~ -0.03 to -0.04)")
    print(f"[deviation] target preshock (≈1.78) panel_a mean = "
          f"{summary['panel_a_at_target']['panel_a_mean']:.4f} (paper ≈ -0.13)")


if __name__ == "__main__":
    main()