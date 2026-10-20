"""run_strategy_graph.py -- 任务 B: Section IV.C 策略图.

读取 representative.npz, 跑:
  - 策略图统计 (跨会话): 全收敛占比、few-nonconv 占比、路径长度均值/最大;
  - 图 8: 单个 1-Sym 会话的有向图 (吸收节点方形, 圆形其余; 大小=betweenness,
    深浅=利润增益 Δ);
  - 图 9: 相图 (节点坐标=(p1,p2), 箭头表示价格变化方向; 标注 p^N、p^M).
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
from strategy_graph import (  # noqa: E402
    build_networkx_graph,
    pick_representative_session,
    strategy_graph_stats,
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


def _plot_figure8(graph_pack: dict, out_path: str) -> None:
    import networkx as nx  # local to keep top-level import clean
    G = graph_pack["graph"]
    node_gain = graph_pack["node_gain"]
    M = G.number_of_nodes()
    grid_prices = np.array([G.nodes[s]["price_pair"] for s in range(M)])

    # 找吸收节点 (出边指向自身: next_state == s)
    next_state = graph_pack["next_state"]
    absorbing_set = [int(s) for s in range(M) if next_state[s] == s]
    # 若没有吸收节点, 找连通分量中最短周期
    if not absorbing_set:
        # 找图中最短环: 从每个节点出发, 跟踪到第一次回到自己
        for start in range(M):
            s = s
            for _ in range(30):
                s = int(next_state[s])
                if s == start:
                    absorbing_set = [start]
                    break
            if absorbing_set:
                break
    absorbing_set = set(absorbing_set)

    # betweenness centrality
    bc = nx.betweenness_centrality(G, normalized=False)
    sizes = np.array([50 + 600 * bc.get(s, 0) for s in range(M)])
    colors = np.clip(node_gain, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    # 布局: 用 grid 价对 (p1, p2) 作为坐标 (论文 Figure 8 看起来是分轴拓扑)
    pos = {s: (float(G.nodes[s]["p1"]), float(G.nodes[s]["p2"])) for s in range(M)}
    # 节点
    circles = [s for s in range(M) if s not in absorbing_set]
    squares = [s for s in range(M) if s in absorbing_set]

    if circles:
        xs = [pos[s][0] for s in circles]
        ys = [pos[s][1] for s in circles]
        ss = [sizes[s] for s in circles]
        cc = [colors[s] for s in circles]
        ax.scatter(xs, ys, s=ss, c=cc, cmap="Greys", vmin=0.0, vmax=1.0,
                   edgecolors="black", linewidths=0.4, marker="o", alpha=0.95)
    if squares:
        xs = [pos[s][0] for s in squares]
        ys = [pos[s][1] for s in squares]
        ss = [sizes[s] for s in squares]
        cc = [colors[s] for s in squares]
        ax.scatter(xs, ys, s=ss, c=cc, cmap="Greys", vmin=0.0, vmax=1.0,
                   edgecolors="black", linewidths=0.6, marker="s")

    # 边 (短一些, 便于辨认)
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # 截断到 0.6 倍长度
        dx, dy = x2 - x1, y2 - y1
        ax.annotate("", xy=(x1 + 0.7 * dx, y1 + 0.7 * dy),
                    xytext=(x1 + 0.3 * dx, y1 + 0.3 * dy),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.35,
                                    shrinkA=2, shrinkB=2))

    ax.set_xlabel("Price of firm 1")
    ax.set_ylabel("Price of firm 2")
    ax.set_title("Figure 8: Directed graph of limit strategies (single session)")
    ax.set_xlim(1.4, 2.0)
    ax.set_ylim(1.4, 2.0)
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Profit gain (Δ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_figure9(graph_pack: dict, p_nash: float, p_mono: float, out_path: str) -> None:
    G = graph_pack["graph"]
    next_state = graph_pack["next_state"]
    M = G.number_of_nodes()

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = {s: (float(G.nodes[s]["p1"]), float(G.nodes[s]["p2"])) for s in range(M)}

    # 找吸收节点
    absorbing_set = [int(s) for s in range(M) if next_state[s] == s]
    if not absorbing_set:
        for start in range(M):
            s = start
            for _ in range(30):
                s = int(next_state[s])
                if s == start:
                    absorbing_set = [start]
                    break
            if absorbing_set:
                break

    # 画箭头: 从 (p1,p2) 指向 0.5*(p1_next + p1_prev) - 短一些便于看清
    for s in range(M):
        x1, y1 = pos[s]
        s_next = int(next_state[s])
        x2, y2 = pos[s_next]
        dx, dy = x2 - x1, y2 - y1
        # 论文做法: "do not extend as far as the next node" -- 取约 0.5
        target_x = x1 + 0.5 * dx
        target_y = y1 + 0.5 * dy
        ax.annotate("", xy=(target_x, target_y), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="C0", lw=0.5,
                                    shrinkA=2, shrinkB=2))

    # 画节点
    xs = [pos[s][0] for s in range(M)]
    ys = [pos[s][1] for s in range(M)]
    ax.scatter(xs, ys, s=18, c="C3", edgecolors="black", linewidths=0.3, alpha=0.6,
               label="States")

    # 吸收节点用方形
    if absorbing_set:
        abs_xs = [pos[s][0] for s in absorbing_set]
        abs_ys = [pos[s][1] for s in absorbing_set]
        ax.scatter(abs_xs, abs_ys, s=80, c="C3", edgecolors="black",
                   linewidths=0.6, marker="s", label="Absorbing state")

    # p^N / p^M 标注
    for px, py, label, color in [(p_nash, p_nash, r"$p^N$", "black"),
                                  (p_mono, p_mono, r"$p^M$", "black")]:
        ax.plot(px, py, "r+", markersize=12, mew=1.5)
        ax.annotate(label, xy=(px, py), xytext=(5, 5),
                    textcoords="offset points", fontsize=11)

    ax.set_xlabel("Price of firm 1")
    ax.set_ylabel("Price of firm 2")
    ax.set_title("Figure 9: Phase diagram of limit strategies (single session)")
    ax.set_xlim(1.4, 2.0)
    ax.set_ylim(1.4, 2.0)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(THIS_DIR, "results", "representative.npz"))
    parser.add_argument("--out-prefix", default=os.path.join(THIS_DIR, "results"))
    parser.add_argument("--fig-prefix", default=os.path.join(THIS_DIR, "figures"))
    args = parser.parse_args(argv)

    os.makedirs(args.out_prefix, exist_ok=True)
    os.makedirs(args.fig_prefix, exist_ok=True)

    print(f"[strategy_graph] loading {args.input}")
    d = np.load(args.input)
    res = _to_session_result(d)
    env = res.env
    grid = env["grid"]
    m = len(grid)
    n_conv = int(res.converged.sum())
    print(f"[strategy_graph] {res.converged.shape[0]} sessions, {n_conv} converged")

    t0 = time.time()

    # ---- 跨会话统计 ---- #
    print("[strategy_graph] computing cross-session stats")
    stats = strategy_graph_stats(res, m=m, n=2)

    out_stats = {
        "paper_targets": {
            "frac_all_converge": 0.92,
            "frac_few_nonconv": 0.98,
            "mean_path_length": 6.0,
            "max_path_length": 18.0,
        },
        "n_sessions": stats.n_sessions,
        "n_converged": n_conv,
        "n_sessions_all_converge": stats.n_sessions_all_converge,
        "n_sessions_few_nonconv": stats.n_sessions_few_nonconv,
        "frac_all_converge": stats.frac_all_converge,
        "frac_few_nonconv": stats.frac_few_nonconv,
        "mean_path_length_overall": stats.mean_path_length_overall,
        "mean_path_length_per_session_mean":
            float(np.nanmean(stats.mean_path_length_per_session))
            if any(np.isfinite(x) for x in stats.mean_path_length_per_session) else float("nan"),
        "max_path_length": stats.max_path_length,
        "n_total_nodes_per_session": stats.n_total_nodes,
        "per_session_nonconv_nodes": stats.per_session_nonconv_nodes,
        "per_session_absorbing_size": stats.per_session_absorbing_size,
        "per_session_cycle_type": stats.per_session_cycle_type,
        "p_nash": float(env["p_nash"]),
        "p_mono": float(env["p_mono"]),
    }
    with open(os.path.join(args.out_prefix, "strategy_graph_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(out_stats, f, ensure_ascii=False, indent=2)

    print(f"[strategy_graph] all-converge fraction = {stats.frac_all_converge:.3f} (paper 0.92)")
    print(f"[strategy_graph] few-nonconv fraction  = {stats.frac_few_nonconv:.3f} (paper 0.98)")
    print(f"[strategy_graph] mean path length      = {stats.mean_path_length_overall:.2f} (paper 6)")
    print(f"[strategy_graph] max path length       = {stats.max_path_length} (paper 18)")

    # ---- Figure 8 & 9 ---- #
    print("[strategy_graph] selecting representative session")
    idx = pick_representative_session(res, prefer_cycle_type=0,
                                       prefer_close_to_p98=True)
    print(f"[strategy_graph] picked session {idx}; longrun_price={res.longrun_prices[idx]}")
    limit_row = res.limit_actions[idx]
    pack = build_networkx_graph(limit_row, env)

    print("[strategy_graph] drawing Figure 8")
    _plot_figure8(pack, os.path.join(args.fig_prefix, "fig8_strategy_graph.png"))
    print("[strategy_graph] drawing Figure 9")
    _plot_figure9(pack, float(env["p_nash"]), float(env["p_mono"]),
                   os.path.join(args.fig_prefix, "fig9_phase_portrait.png"))

    # ---- JSON addendum: which session was used ---- #
    out_stats["figure8_9_session"] = int(idx)
    out_stats["figure8_9_longrun_prices"] = [float(p) for p in res.longrun_prices[idx]]
    with open(os.path.join(args.out_prefix, "strategy_graph_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(out_stats, f, ensure_ascii=False, indent=2)

    print(f"[strategy_graph] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()