"""run_grid.py -- (alpha, beta) 参数网格, 对应论文图 1 / 图 2 / 图 7。

论文用 100x100 网格 (每个点 1000 个会话) —— 那是 1e10 会话量级, 本机无法承受。
这里默认 20x20 网格 x 每点 6 个会话, 可用命令行放大。网格区间与论文一致:
  alpha in [0.025, 0.25], beta in [0, 2e-5]。

每个网格点统计:
  * 平均利润增益 Delta (仅收敛会话)                -> 图 1
  * 收敛到 Nash 均衡的会话比例 (在轨最优回应)      -> 图 2
  * 非偏离方在 tau=2 的平均相对降价 (惩罚强度)     -> 图 7
  * Q-loss (在轨 / 全状态)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analysis import evaluate_sessions_vec          # noqa: E402
from core import MarketSpec                          # noqa: E402
from engine import SimConfig                         # noqa: E402
from parallel import DEFAULT_WORKERS, run_configs    # noqa: E402

FIG_DIR = "figures"


def heatmap(Z, alphas, betas, title, path, cbar_label, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                   extent=[betas[0] * 1e5, betas[-1] * 1e5, alphas[0], alphas[-1]],
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$\beta \times 10^{5}$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=20, help="beta 方向网格数")
    ap.add_argument("--ny", type=int, default=20, help="alpha 方向网格数")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--periods", type=int, default=3_000_000)
    ap.add_argument("--patience", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", type=int, default=0, help="第几个分片 (0 起)")
    ap.add_argument("--nshard", type=int, default=1, help="总���数")
    ap.add_argument("--merge", action="store_true", help="合并各分片 CSV 并出图")
    args = ap.parse_args()

    if args.merge:
        return merge_shards(args.tag, args.nx, args.ny)

    tag = args.tag or f"{args.nx}x{args.ny}x{args.sessions}"
    os.makedirs("results", exist_ok=True)

    alphas = np.linspace(0.025, 0.25, args.ny)
    all_betas = np.linspace(0.0, 2e-5, args.nx)
    # 按分片切分 beta 轴 (每个分片写独立 CSV, 最后 --merge 汇总)
    bsel = np.array_split(np.arange(args.nx), args.nshard)[args.shard]
    betas = all_betas[bsel]
    cfgs = []
    for a in alphas:
        for b in betas:
            cfgs.append(SimConfig(spec=MarketSpec(), alpha=float(a), beta=float(b),
                                  patience=args.patience, max_periods=args.periods,
                                  profit_window=1000))
    suffix = "" if args.nshard == 1 else f"_shard{args.shard}"
    print(f"[grid] shard {args.shard}/{args.nshard}: {args.ny} alphas x {len(betas)} betas "
          f"x {args.sessions} sessions = {len(cfgs) * args.sessions} sessions",
          flush=True)

    t0 = time.perf_counter()
    results = run_configs(cfgs, args.sessions, n_workers=1, base_seed=77000 + args.shard)
    print(f"[grid] shard {args.shard} simulation finished in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    env = results[0].env
    grid = env["grid"]
    pi_tab = env["pi_table"]
    rows = []
    for k, res in enumerate(results):
        conv = res.converged
        ev = evaluate_sessions_vec(res.limit_actions, pi_tab, grid, delta=0.95,
                                   onpath=res.onpath_states)
        rows.append({
            "alpha": cfgs[k].alpha, "beta": cfgs[k].beta,
            "converged": int(conv.sum()), "n_sessions": res.limit_actions.shape[0],
            "t_conv_median": float(np.median(res.t_conv[conv])) if conv.any() else -1.0,
            "delta": float(np.nanmean(res.delta()[conv])) if conv.any() else np.nan,
            "nash": float(np.mean(ev["nash_onpath"][conv])) if conv.any() else np.nan,
            "nash_either": float(np.mean(ev["either_best_onpath"][conv])) if conv.any() else np.nan,
            "qloss_onpath": float(np.nanmean(ev["qloss_onpath"][conv])) if conv.any() else np.nan,
            "qloss_all": float(np.nanmean(ev["qloss_all"][conv])) if conv.any() else np.nan,
            "punish_tau2": float(np.nanmean(ev["punish_tau2"][conv])) if conv.any() else np.nan,
        })
        if (k + 1) % 20 == 0:
            print(f"[grid]   evaluated {k + 1}/{len(cfgs)}", flush=True)

    csv_path = f"results/grid_{tag}{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[grid] wrote {csv_path} ({os.path.getsize(csv_path):,} bytes)", flush=True)
    if args.nshard > 1:
        print("[grid] 分片完成, 全部跑完后用 --merge 汇总出图", flush=True)
        return


def merge_shards(tag: str, nx: int, ny: int) -> None:
    """合并各分片 CSV -> 完整 CSV + 图 1/2/7 + 汇总 JSON。"""
    rows = []
    for k in range(0, 100):
        p = f"results/grid_{tag}_shard{k}.csv"
        if not os.path.exists(p):
            if k == 0:
                raise SystemExit(f"找不到 {p}")
            break
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    for r in rows:
        for k in ("alpha", "beta", "t_conv_median", "delta", "nash", "nash_either",
                  "qloss_onpath", "qloss_all", "punish_tau2"):
            r[k] = float(r[k])
        r["converged"] = int(r["converged"])
        r["n_sessions"] = int(r["n_sessions"])
    alphas = np.linspace(0.025, 0.25, ny)
    betas = np.linspace(0.0, 2e-5, nx)
    mats = {k: np.full((ny, nx), np.nan) for k in ("delta", "nash", "punish_tau2")}
    for r in rows:
        i = int(np.argmin(np.abs(alphas - r["alpha"])))
        j = int(np.argmin(np.abs(betas - r["beta"])))
        for k in mats:
            mats[k][i, j] = r[k]
    csv_path = f"results/grid_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    d = mats["delta"].ravel()
    ok = ~np.isnan(d)
    d, nq, pu = d[ok], mats["nash"].ravel()[ok], mats["punish_tau2"].ravel()[ok]
    # 论文口径: 惩罚强度 = 1 - (tau=2 相对价格), 数值越大惩罚越狠 (图 7 色标 0.84~0.96)
    intensity = -pu
    summary = {
        "grid": [ny, nx], "n_points": len(rows),
        "sessions_per_point": int(rows[0]["n_sessions"]),
        "alpha_range": [0.025, 0.25], "beta_range": [0.0, 2e-5],
        "delta_mean": float(d.mean()), "delta_min": float(d.min()),
        "delta_max": float(d.max()), "nash_mean": float(nq.mean()),
        "punish_multiplier_mean": float(1.0 + pu.mean()),
        "punish_multiplier_min": float(1.0 + pu.min()),
        "punish_multiplier_max": float(1.0 + pu.max()),
        "corr_delta_nash": (float(np.corrcoef(d, nq)[0, 1])
                            if len(set(nq.tolist())) > 1 else None),
        "corr_delta_punishment_intensity": (float(np.corrcoef(d, intensity)[0, 1])
                                            if len(set(intensity.tolist())) > 1 else None),
        "paper_targets": {"delta_range": [0.70, 0.90], "corr_delta_nash": 0.12,
                          "corr_delta_punish": 0.762,
                          "fig7_colour_range": [0.84, 0.96]},
    }
    with open(f"results/grid_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[grid] merged summary:", json.dumps(summary, ensure_ascii=False, indent=2))
    outs = [
        heatmap(mats["delta"], alphas, betas, "Figure 1. Average Profit Gain $\\Delta$",
                f"{FIG_DIR}/fig1_profit_gain_grid.png", r"$\Delta$", vmin=0.70, vmax=0.90),
        heatmap(mats["nash"], alphas, betas,
                "Figure 2. Fraction of Sessions Converging to a Nash Equilibrium",
                f"{FIG_DIR}/fig2_nash_frequency_grid.png", "fraction", vmin=0.0, vmax=0.8),
        heatmap(1.0 + mats["punish_tau2"], alphas, betas,
                "Figure 7. Price of Non-deviating Agent at $\\tau=2$ "
                "(relative to pre-shock price)",
                f"{FIG_DIR}/fig7_punishment_intensity_grid.png", "price ratio",
                vmin=0.84, vmax=0.96),
    ]
    for p in outs + [csv_path, f"results/grid_{tag}_summary.json"]:
        print(f"[grid] wrote {p} ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()