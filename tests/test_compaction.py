"""回归测试: 会话压缩 (行剔除) 路径。

robust-timescale 队友报告的 bug: 已收敛会话被剔除后, per-row 系数数组
(alpha/1-alpha/delta*alpha) 仍按初始 N 长度, 导致广播失败。
本测试用"少量会话 + 极小 patience + 带 history"强制触发压缩路径。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import MarketSpec                                   # noqa: E402
from engine import SimConfig, build_env, run_batch             # noqa: E402


def case(name: str, cfg: SimConfig, sessions: int) -> None:
    env = build_env(cfg)
    res = run_batch([cfg], sessions, seed=11, env=env)
    d = res.delta()
    conv = res.converged
    print(f"[{name}] sessions={sessions} converged={int(conv.sum())}/{sessions} "
          f"t_conv={res.t_conv[conv][:6]} Delta(mean)={np.nanmean(d):.3f} "
          f"limit_actions={res.limit_actions.shape} "
          f"history={'yes' if res.history else 'no'}")
    assert res.limit_actions.shape[0] == sessions
    assert np.isfinite(res.pi_final).all(), "存在未被冻结的会话"


def main() -> None:
    # 1) 队友报告的最小复现
    case("tiny-patience", SimConfig(patience=500, max_periods=60_000), 3)
    # 2) 压缩 + 多实验点混合 (alpha 不同 -> 系数数组必须按行切片)
    cfgs = [SimConfig(alpha=a, patience=800, max_periods=60_000) for a in (0.05, 0.15, 0.25)]
    env = build_env(cfgs[0])
    res = run_batch(cfgs, 4, seed=5, env=env)
    print(f"[mixed-points] N={res.limit_actions.shape[0]} "
          f"converged={int(res.converged.sum())} "
          f"alpha per row={np.unique(res.meta['alpha'])} "
          f"Delta(mean)={np.nanmean(res.delta()):.3f}")
    # 3) 压缩 + 学习曲线记录
    case("with-history", SimConfig(patience=700, max_periods=60_000,
                                   record_every=2000, fixed_periods=None), 5)
    # 4) n=3 (更大状态空间) 的压缩路径
    case("n=3", SimConfig(spec=MarketSpec(n=3, quality=(2., 2., 2.), cost=(1., 1., 1.)),
                          patience=2000, max_periods=40_000), 4)
    print("\nALL COMPACTION CASES OK")


if __name__ == "__main__":
    main()