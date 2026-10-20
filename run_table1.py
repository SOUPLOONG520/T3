"""run_table1.py -- 用代表性实验产出论文表 1 的描述统计 + 最优回应检验。

表 1 列: 按周期长度分组的 频率 / 平均利润增益 / 标准差 / Nash 频率 / Q-loss(在轨) / Q-loss(全状态)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analysis import evaluate_sessions_vec                     # noqa: E402
from engine import (CYCLE_2, CYCLE_CONST_ASYM, CYCLE_CONST_SYM,   # noqa: E402
                    CYCLE_GE3, CYCLE_NAMES)

GROUPS = [CYCLE_CONST_SYM, CYCLE_CONST_ASYM, CYCLE_2, CYCLE_GE3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/representative.npz")
    ap.add_argument("--out", default="results/table1_stats.json")
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    z = np.load(args.npz)
    grid, pi_tab = z["grid"], z["pi_table"]
    delta_f = float(z["delta"]) if "delta" in z else 0.95
    if "delta_grid" in z.files:
        d = z["delta_grid"]
    elif "pi_final" in z.files:
        d = ((z["pi_final"] - float(z["pi_nash"]))
             / (float(z["pi_mono"]) - float(z["pi_nash"])))       # 式 (9)
    else:
        d = (z["pi_final"] - float(z["pi_nash"])) / (
            float(z["pi_mono"]) - float(z["pi_nash"]))
    conv = z["converged"].astype(bool)
    cyc = z["cycle_type"].astype(int)
    n = min(len(conv), args.max_sessions) if args.max_sessions else len(conv)
    conv, cyc, delta_arr = conv[:n], cyc[:n], d[:n]
    idx = np.arange(n)

    ev = evaluate_sessions_vec(z["limit_actions"][:n], pi_tab, grid, delta=delta_f,
                               onpath=z["onpath_states"][:n])

    table = {}
    for g in GROUPS + ["all"]:
        if g == "all":
            sel = idx
        else:
            sel = idx[cyc == g]
        if len(sel) == 0:
            continue
        table[CYCLE_NAMES.get(g, "all")] = {
            "n": int(len(sel)),
            "frequency": float(len(sel) / n),
            "delta_mean": float(np.nanmean(delta_arr[sel])),
            "delta_sd": float(np.nanstd(delta_arr[sel])),
            "freq_nash_onpath": float(np.mean(ev["nash_onpath"][sel])),
            "freq_either_best": float(np.mean(ev["either_best_onpath"][sel])),
            "qloss_onpath_mean": float(np.mean(ev["qloss_onpath"][sel])),
            "qloss_onpath_sd": float(np.std(ev["qloss_onpath"][sel])),
            "qloss_all_mean": float(np.mean(ev["qloss_all"][sel])),
            "qloss_all_sd": float(np.std(ev["qloss_all"][sel])),
        }
    # 只在 Nash 会话上的统计 (论文最后一列)
    nash_sel = idx[ev["nash_onpath"]]
    if len(nash_sel):
        table["nash_only"] = {
            "n": int(len(nash_sel)),
            "frequency": float(len(nash_sel) / n),
            "delta_mean": float(np.nanmean(delta_arr[nash_sel])),
            "delta_sd": float(np.nanstd(delta_arr[nash_sel])),
            "qloss_onpath_mean": float(np.mean(ev["qloss_onpath"][nash_sel])),
            "qloss_all_mean": float(np.mean(ev["qloss_all"][nash_sel])),
        }

    out = {
        "source": args.npz, "n_sessions": int(n), "delta": delta_f,
        "converged": int(conv.sum()),
        "delta_mean_all": float(np.nanmean(delta_arr)),
        "corr_delta_nash": (float(np.corrcoef(delta_arr, ev["nash_onpath"].astype(float))[0, 1])
                            if ev["nash_onpath"].any() and ev["nash_onpath"].any() is False
                            else None),
        "paper_targets": {"1-Sym": 0.277, "1-Asym": 0.366, "2": 0.238, ">=3": 0.119,
                          "avg_delta": 0.849, "freq_nash": 0.505,
                          "qloss_onpath": 0.001, "qloss_all": 0.018},
        "table": table,
        "mean_qloss_onpath": float(np.mean(ev["qloss_onpath"])),
        "mean_qloss_all": float(np.mean(ev["qloss_all"])),
        "freq_nash_onpath": float(np.mean(ev["nash_onpath"])),
        "freq_either_best_onpath": float(np.mean(ev["either_best_onpath"])),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[table1] wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()