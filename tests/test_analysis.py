"""检验 analysis.evaluate_sessions_vec 与逐会话标量版是否一致。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis import evaluate_sessions_vec, session_best_response   # noqa: E402
from core import MarketSpec                                          # noqa: E402
from engine import SimConfig, build_env, run_batch                   # noqa: E402


def main() -> None:
    cfg = SimConfig(spec=MarketSpec(), alpha=0.15, beta=4e-6,
                    fixed_periods=6000, profit_window=200)
    env = build_env(cfg)
    res = run_batch([cfg], 6, seed=42, env=env)

    vec = evaluate_sessions_vec(res.limit_actions, env["pi_table"], env["grid"],
                                delta=0.95, onpath=res.onpath_states)
    W = min(res.onpath_states.shape[1], 200)
    sc_nash, sc_ql_on, sc_ql_all, sc_pun = [], [], [], []
    for r in range(res.limit_actions.shape[0]):
        d = session_best_response(res.limit_actions[r], env, 0.95,
                                  onpath_row=res.onpath_states[r, -W:])
        sc_nash.append(bool(np.all(d["is_best_onpath"])))
        sc_ql_on.append(float(np.mean(d["qloss_onpath"])))
        sc_ql_all.append(float(np.mean(d["qloss_all"])))
    print("scalar nash_onpath :", sc_nash)
    print("vector nash_onpath :", vec["nash_onpath"].tolist())
    print("scalar qloss_onpath:", np.round(sc_ql_on, 8).tolist())
    print("vector qloss_onpath:", np.round(vec["qloss_onpath"], 8).tolist())
    print("scalar qloss_all   :", np.round(sc_ql_all, 8).tolist())
    print("vector qloss_all   :", np.round(vec["qloss_all"], 8).tolist())
    print("punish tau=2 (vec) :", np.round(vec["punish_tau2"], 4).tolist())

    ok = (sc_nash == vec["nash_onpath"].tolist()
          and np.allclose(sc_ql_on, vec["qloss_onpath"], atol=1e-6)
          and np.allclose(sc_ql_all, vec["qloss_all"], atol=1e-6))
    print("\nVERDICT:", "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()