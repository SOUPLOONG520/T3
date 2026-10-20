"""run_robustness.py -- 论文第 V 节稳健性 + 图 3 δ 扫描 + k=0 对照 的入口。

支持命令行 --only 指定子实验，便于分批完成：
  --only analytic              只跑解析校验（成本不对称 Nash 份额）
  --only delta,k0              δ 扫描 + k=0 对照
  --only nfirms                n=2/3/4 + n=3 调 β
  --only cost                  成本不对称（表 4）
  --only demand                随机需求
  --only mu                    μ 变化
  --only init                  初始化
  --only actionset             动作集
  --only all                   全部（默认）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "src"))

from robustness import (  # noqa: E402
    analytic_nash_shares,
    plot_fig3,
    run_action_set,
    run_cost_asymmetry,
    run_delta_sweep,
    run_demand_shock,
    run_initialization,
    run_memoryless,
    run_mu_variation,
    run_n_firms,
    save_csv,
    save_json,
)

FIGURES = os.path.join(THIS_DIR, "figures")

RESULTS = os.path.join(THIS_DIR, "results")


def _do_analytic() -> None:
    out = analytic_nash_shares()
    save_json({"analytic_nash_shares": out},
              os.path.join(RESULTS, "robustness_analytic_nash.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_analytic_nash.csv"))
    print("[analytic] Nash shares written.")


def _do_delta_k0(args) -> None:
    t0 = time.time()
    out_delta = run_delta_sweep(n_sessions=args.sessions_delta,
                                n_workers=args.n_workers,
                                fixed_periods=args.fixed_periods)
    save_json(out_delta, os.path.join(RESULTS, "robustness_delta.json"))
    save_csv(out_delta["rows"], os.path.join(RESULTS, "robustness_delta.csv"))
    print(f"[delta] {len(out_delta['deltas'])} deltas × "
          f"{out_delta['sessions_per_delta']} sessions × "
          f"{out_delta['fixed_periods']} periods in {time.time()-t0:.1f}s")

    t0 = time.time()
    out_k0 = run_memoryless(n_sessions=args.sessions_delta,
                            n_workers=args.n_workers,
                            fixed_periods=args.fixed_periods)
    save_json(out_k0, os.path.join(RESULTS, "robustness_k0.json"))
    print(f"[k=0] {out_k0['n']} sessions in {time.time()-t0:.1f}s; "
          f"mean Δ={out_k0['mean_delta']:.3f} ({out_k0['method']})")

    os.makedirs(FIGURES, exist_ok=True)
    plot_fig3(out_delta["rows"], out_k0,
              os.path.join(FIGURES, "fig3_delta_vs_discount.png"))
    print(f"[fig3] written to {os.path.join(FIGURES, 'fig3_delta_vs_discount.png')}")


def _do_nfirms(args) -> None:
    t0 = time.time()
    out = run_n_firms(
        n_sessions_by_n={2: args.sessions_n2, 3: args.sessions_n3,
                          4: args.sessions_n4},
        n_workers=args.n_workers,
        fixed_periods=args.fixed_periods,
    )
    save_json(out, os.path.join(RESULTS, "robustness_nfirms.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_nfirms.csv"))
    print(f"[nfirms] done in {time.time()-t0:.1f}s")


def _do_cost(args) -> None:
    t0 = time.time()
    out = run_cost_asymmetry(n_sessions=args.sessions_cost,
                             n_workers=args.n_workers,
                             fixed_periods=args.fixed_periods)
    save_json(out, os.path.join(RESULTS, "robustness_cost.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "table4_cost_asymmetry.csv"))
    print(f"[cost] 6 cases × {args.sessions_cost} sessions in {time.time()-t0:.1f}s")


def _do_demand(args) -> None:
    t0 = time.time()
    out = run_demand_shock(n_sessions=args.sessions_demand,
                            n_workers=args.n_workers,
                            fixed_periods=args.fixed_periods)
    save_json(out, os.path.join(RESULTS, "robustness_demand.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_demand.csv"))
    print(f"[demand] 2 shocks × {args.sessions_demand} sessions in {time.time()-t0:.1f}s")


def _do_mu(args) -> None:
    t0 = time.time()
    out = run_mu_variation(n_sessions=args.sessions_mu,
                            n_workers=args.n_workers,
                            fixed_periods=args.fixed_periods)
    save_json(out, os.path.join(RESULTS, "robustness_mu.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_mu.csv"))
    print(f"[mu] {len(out['rows'])} cases × {args.sessions_mu} sessions "
          f"in {time.time()-t0:.1f}s")


def _do_init(args) -> None:
    t0 = time.time()
    out = run_initialization(n_sessions=args.sessions_init,
                              n_workers=args.n_workers,
                              fixed_periods=args.fixed_periods)
    save_json(out, os.path.join(RESULTS, "robustness_init.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_init.csv"))
    print(f"[init] {len(out['rows'])} cases × {args.sessions_init} sessions "
          f"in {time.time()-t0:.1f}s")


def _do_actionset(args) -> None:
    t0 = time.time()
    out = run_action_set(n_sessions_normal=args.sessions_act,
                          n_sessions_m100=args.sessions_act_m100,
                          n_workers=args.n_workers,
                          fixed_periods=args.fixed_periods)
    save_json(out, os.path.join(RESULTS, "robustness_actionset.json"))
    save_csv(out["rows"], os.path.join(RESULTS, "robustness_actionset.csv"))
    print(f"[actionset] {len(out['rows'])} cases in {time.time()-t0:.1f}s")


DISPATCH = {
    "analytic": _do_analytic,
    "delta": lambda a: _do_delta_k0(a),       # alias 也归并到 delta+k0
    "k0": lambda a: _do_delta_k0(a),
    "delta,k0": lambda a: _do_delta_k0(a),
    "nfirms": _do_nfirms,
    "cost": _do_cost,
    "demand": _do_demand,
    "mu": _do_mu,
    "init": _do_init,
    "actionset": _do_actionset,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        help=("comma-separated subset of {analytic,delta,k0,nfirms,"
                              "cost,demand,mu,init,actionset,all}"))
    # Lead 协调（2026-10-20 19:08）：单进程；会话数比周期数重要（R=80 时 ~300K/s）
    parser.add_argument("--sessions_delta", type=int, default=80)
    parser.add_argument("--sessions_n2", type=int, default=80)
    parser.add_argument("--sessions_n3", type=int, default=40)
    parser.add_argument("--sessions_n4", type=int, default=25)
    parser.add_argument("--sessions_cost", type=int, default=80)
    parser.add_argument("--sessions_demand", type=int, default=80)
    parser.add_argument("--sessions_mu", type=int, default=80)
    parser.add_argument("--sessions_init", type=int, default=80)
    parser.add_argument("--sessions_act", type=int, default=80)
    parser.add_argument("--sessions_act_m100", type=int, default=15)
    parser.add_argument("--fixed_periods", type=int, default=600_000)
    parser.add_argument("--n_workers", type=int, default=1,
                        help="沙箱内进程池不可用，请保持 1")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    targets = (
        list(DISPATCH.keys()) if args.only == "all"
        else [t.strip() for t in args.only.split(",") if t.strip()]
    )
    for t in targets:
        if t not in DISPATCH:
            print(f"[skip] unknown target '{t}'")
            continue
        print(f"\n=== Running {t} ===")
        DISPATCH[t](args)


if __name__ == "__main__":
    main()