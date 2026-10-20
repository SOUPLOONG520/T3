"""run_nash_long.py -- 诊断: 在轨 Nash 频率为何为 0。

论文报告在轨 Nash 频率 0.505, 但其会话上限是 10 亿期。本工程只跑到 2.5e6~3e6 期。
本脚本比较不同运行长度下的 (Nash 频率, Q-loss), 检验 "拉长运行是否让学习到的
argmax 收敛到真 Q 的 argmax" 这一解释。

注意: 固定期数模式 (fixed_periods) 下会话不会提前冻结, 全部跑满。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analysis import evaluate_sessions_vec      # noqa: E402
from core import MarketSpec                      # noqa: E402
from engine import SimConfig                     # noqa: E402
from parallel import run_config                  # noqa: E402


def one(periods: int, sessions: int) -> dict:
    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    fixed_periods=periods, profit_window=1000)
    res = run_config(cfg, sessions, n_workers=1, base_seed=31337)
    ev = evaluate_sessions_vec(res.limit_actions, res.env["pi_table"],
                               res.env["grid"], delta=0.95,
                               onpath=res.onpath_states)
    eps_end = float(np.exp(-4e-6 * periods))
    d = res.delta()
    return {
        "periods": periods,
        "sessions": sessions,
        "eps_at_end": eps_end,
        "delta_mean": float(np.nanmean(d)),
        "nash_onpath": float(np.mean(ev["nash_onpath"])),
        "either_best_onpath": float(np.mean(ev["either_best_onpath"])),
        "qloss_onpath": float(np.mean(ev["qloss_onpath"])),
        "qloss_all": float(np.mean(ev["qloss_all"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--periods", type=int, nargs="+", default=[1_000_000, 3_000_000, 8_000_000])
    ap.add_argument("--out", default="results/nash_vs_length.json")
    args = ap.parse_args()
    rows = []
    for p in args.periods:
        r = one(p, args.sessions)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "paper": {"freq_nash_onpath": 0.505,
                                           "qloss_onpath": 0.001,
                                           "max_periods": 1_000_000_000}},
                  f, ensure_ascii=False, indent=2)
    print(f"[nash-long] wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()