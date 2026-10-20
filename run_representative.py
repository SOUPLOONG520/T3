"""run_representative.py -- 生成 representative.npz (Section IV 的基础数据)

代表性实验 alpha=0.15, beta=4e-6, 150 会话. 关键点:

* 默认 `min_rows=1` (per-session chunk), 让每个会话有独立的 rng seed,
  避免全部会话在同一 rng 下同步走向同一吸引子;
* `REP_PATIENCE` (默认 200_000) 是 argmax 连续不变期数的阈值;
* `REP_MAX_PERIODS` (默认 5_000_000) 是硬上限.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "src"))

from core import MarketSpec  # noqa: E402
from engine import SimConfig  # noqa: E402
from parallel import run_config  # noqa: E402


def main() -> None:
    cfg = SimConfig(
        spec=MarketSpec(),
        alpha=0.15,
        beta=4e-6,
        patience=int(os.environ.get("REP_PATIENCE", "200_000")),
        max_periods=int(os.environ.get("REP_MAX_PERIODS", "5_000_000")),
        profit_window=1000,
    )
    n_sessions = int(os.environ.get("REP_SESSIONS", "150"))
    n_workers = int(os.environ.get("REP_WORKERS", "20"))
    min_rows = int(os.environ.get("REP_MIN_ROWS", "200"))
    out = os.environ.get("REP_OUT", os.path.join(THIS_DIR, "results", "representative.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    return_q = bool(int(os.environ.get("REP_RETURN_Q", "0")))
    res = run_config(cfg, n_sessions, n_workers=n_workers, min_rows=min_rows,
                     return_q=return_q)
    dt = time.time() - t0

    save_kwargs = dict(
        converged=res.converged,
        t_conv=res.t_conv,
        pi_final=res.pi_final,
        longrun_actions=res.longrun_actions,
        longrun_prices=res.longrun_prices,
        cycle_type=res.cycle_type,
        limit_actions=res.limit_actions,
        onpath_states=res.onpath_states,
        grid=res.env["grid"],
        pi_table=res.env["pi_table"],
        pi_nash=res.env["pi_nash"],
        pi_mono=res.env["pi_mono"],
        p_nash=res.env["p_nash"],
        p_mono=res.env["p_mono"],
        alpha=0.15,
        beta=4e-6,
        delta=0.95,
        m=15,
        n=2,
        k=1,
        M=225,
        n_sessions=n_sessions,
        min_rows=min_rows,
        return_q=int(return_q),
        wall_seconds=dt,
    )
    if return_q and getattr(res, "Q_final", None) is not None:
        save_kwargs["Q_final"] = res.Q_final

    np.savez_compressed(out, **save_kwargs)
    n_conv = int(res.converged.sum())
    n_diff_longrun = int(np.unique(res.longrun_prices, axis=0).shape[0])
    print(f"[representative] {n_sessions} sessions in {dt:.1f}s ({dt/60:.1f}min); "
          f"converged={n_conv}/{n_sessions}; delta_mean={res.delta().mean():.3f}; "
          f"unique longrun_prices={n_diff_longrun}")


if __name__ == "__main__":
    main()