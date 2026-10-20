"""strategy_graph.py -- Section IV.C: 策略图 (Strategy graph), Figures 8, 9.

口径:
    - 对每个会话在 M=225 个状态上建有向图: s -> profile_index(limit_actions[:, s], m);
    - 1-cycle: 唯一吸收节点; 周期 L: L 个状态组成环; 不收敛节点: 不在 (环) 闭包内的点;
    - "all nodes lead to long-run" = 每个 225 节点最终都到达吸收集;
    - "fewer than 3 non-convergent nodes" = 不在吸收集闭包内的节点数 < 3.

Profit gain 节点颜色: Δ_node = (π(该节点价格对) - π^N) / (π^M - π^N).
Betweenness centrality: 节点大小.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from core import profile_index


def _state_weights(n: int, m: int) -> np.ndarray:
    w = np.empty(n, dtype=np.int64)
    for j in range(n):
        w[j] = m ** (n - 1 - j)
    return w


def build_adjacency(limit_row: np.ndarray, m: int, n: int = 2) -> np.ndarray:
    """返回 adj (M, M) 布尔邻接矩阵. adj[s, s'] = 1 表示 s -> s'."""
    w = _state_weights(n, m)
    M = limit_row.shape[1]
    adj = np.zeros((M, M), dtype=bool)
    next_state = (limit_row.T.astype(np.int64) @ w).astype(np.int64)
    adj[np.arange(M), next_state] = True
    return adj, next_state


def find_attractor(adj: np.ndarray, start: int, cap: int | None = None) -> list[int]:
    """从 start 出发, 沿 strategy graph 迭代直至回到已访问节点.

    返回该环上所有节点 (按访问顺序). 单收。
    """
    if cap is None:
        cap = adj.shape[0] + 2
    s = int(start)
    attractor = []
    for _ in range(cap):
        if s in attractor:
            return attractor
        attractor.append(s)
        s = int(np.argmax(adj[s]))  # 邻接矩阵每行只有一个 True
    return attractor


def bfs_path_length(adj: np.ndarray, target_set: set[int],
                    max_len: int = 300) -> np.ndarray:
    """逆 BFS: 从每个节点出发到达 target_set 的最短步数.

    Returns: (M,) 数组, target_set 内节点 = 0, 不可达 = -1.
    """
    M = adj.shape[0]
    nxt = np.argmax(adj, axis=1)
    # 前驱表: pred[s] = set of states whose next is s
    pred = [np.where(nxt == s)[0] for s in range(M)]
    dist = -np.ones(M, dtype=np.int64)
    seeds = list(target_set)
    for s in seeds:
        dist[s] = 0
    cur = np.array(seeds, dtype=np.int64)
    for step in range(1, max_len + 1):
        if cur.size == 0:
            break
        # 收集所有前驱, 去重
        new_nodes_list = []
        for s in cur:
            new_nodes_list.extend(pred[int(s)])
        if not new_nodes_list:
            break
        new_nodes = np.unique(np.array(new_nodes_list, dtype=np.int64))
        # 取尚未分配距离的
        new_nodes = new_nodes[dist[new_nodes] < 0]
        if new_nodes.size == 0:
            break
        dist[new_nodes] = step
        cur = new_nodes
    return dist


@dataclass
class GraphStats:
    n_total_nodes: int
    n_sessions: int
    n_sessions_all_converge: int       # 所有节点都能到长期
    n_sessions_few_nonconv: int        # 不收敛节点 < 3
    frac_all_converge: float
    frac_few_nonconv: float
    mean_path_length_overall: float    # 所有 (session, node) 中到达长期步的均值
    mean_path_length_per_session: list[float]
    max_path_length: int               # 全局最大路径长度
    per_session_nonconv_nodes: list[int]
    per_session_absorbing_size: list[int]
    per_session_cycle_type: list[int]


def strategy_graph_stats(res, m: int = 15, n: int = 2,
                         nonconv_threshold: int = 3) -> GraphStats:
    """遍历所有会话, 计算策略图统计量."""
    M = m ** n
    n_sessions = int(res.converged.shape[0])
    n_all_conv = 0
    n_few_nonconv = 0
    path_lengths = []
    per_sess_mean = []
    per_sess_max = []
    per_sess_nonconv = []
    per_sess_absorbing_size = []
    per_sess_cycle = []

    for r in range(n_sessions):
        ct = int(res.cycle_type[r])
        per_sess_cycle.append(ct)
        if not res.converged[r]:
            per_sess_mean.append(np.nan)
            per_sess_max.append(-1)
            per_sess_nonconv.append(M)
            per_sess_absorbing_size.append(0)
            continue

        limit_row = res.limit_actions[r]
        longrun = res.longrun_actions[r]
        adj, nxt = build_adjacency(limit_row, m, n=n)

        # 找到吸收集
        s0 = int(profile_index(longrun, m))
        attractor = find_attractor(adj, s0)
        target_set = set(attractor)
        per_sess_absorbing_size.append(len(target_set))

        # BFS 路径长度
        dist = bfs_path_length(adj, target_set)
        nonconv_count = int(np.sum(dist < 0))
        per_sess_nonconv.append(nonconv_count)

        reachable = dist >= 0
        if reachable.all():
            n_all_conv += 1
        if nonconv_count < nonconv_threshold:
            n_few_nonconv += 1

        # 路径长度均值 / 最大值 (按"可达"节点; 不可达节点排除)
        if reachable.any():
            sess_mean = float(dist[reachable].mean())
            sess_max = int(dist[reachable].max())
            per_sess_mean.append(sess_mean)
            per_sess_max.append(sess_max)
            path_lengths.extend(dist[reachable].tolist())
        else:
            per_sess_mean.append(np.nan)
            per_sess_max.append(-1)

    mean_overall = float(np.mean(path_lengths)) if path_lengths else float('nan')
    global_max = int(max(per_sess_max)) if per_sess_max else -1

    return GraphStats(
        n_total_nodes=M,
        n_sessions=n_sessions,
        n_sessions_all_converge=n_all_conv,
        n_sessions_few_nonconv=n_few_nonconv,
        frac_all_converge=n_all_conv / n_sessions,
        frac_few_nonconv=n_few_nonconv / n_sessions,
        mean_path_length_overall=mean_overall,
        mean_path_length_per_session=per_sess_mean,
        max_path_length=global_max,
        per_session_nonconv_nodes=per_sess_nonconv,
        per_session_absorbing_size=per_sess_absorbing_size,
        per_session_cycle_type=per_sess_cycle,
    )


def pick_representative_session(res, prefer_cycle_type: int = 0,
                               prefer_close_to_p98: bool = True) -> int:
    """挑一个 1-cycle 会话做 Figure 8.

    - 仅在 cycle_type == 0 (1-Sym) 中挑;
    - 若 prefer_close_to_p98: 选 longrun_price 最接近 1.78 的 (论文 Figure 8 案例中
      长期价约为 1.78);
    - 否则选 session 0.
    """
    for r in range(int(res.converged.shape[0])):
        if res.converged[r] and int(res.cycle_type[r]) == prefer_cycle_type:
            pass
    candidates = [r for r in range(int(res.converged.shape[0]))
                  if res.converged[r] and int(res.cycle_type[r]) == prefer_cycle_type]
    if not candidates:
        candidates = [r for r in range(int(res.converged.shape[0])) if res.converged[r]]
    if not candidates:
        return 0
    if not prefer_close_to_p98:
        return candidates[0]
    target = 1.78
    best = min(candidates, key=lambda r: abs(res.longrun_prices[r, 0] - target))
    return best


def build_networkx_graph(limit_row: np.ndarray, env: dict, deviator: int = 0) -> dict:
    """把单个会话的策略图包装成 networkx.DiGraph + 附加节点属性.

    Returns dict: graph, absorbing_set, profits_per_node, node_gain.
    """
    grid = env['grid']
    m = len(grid)
    pi_tab = env['pi_table']
    pi_nash = float(env['pi_nash'])
    pi_mono = float(env['pi_mono'])
    n = 2
    w = _state_weights(n, m)
    M = limit_row.shape[1]

    nxt = (limit_row.T.astype(np.int64) @ w).astype(np.int64)
    G = nx.DiGraph()
    # 节点属性: 坐标 (用 grid 上的价格对) + 利润增益
    node_gain = np.zeros(M, dtype=float)
    for s in range(M):
        # 该节点代表的"价格对"= grid[limit_actions 在 s 处的动作]
        a = limit_row[:, s]
        prices = grid[a]
        profit = pi_tab[profile_index(a.astype(np.int64), m)].mean()
        node_gain[s] = (profit - pi_nash) / (pi_mono - pi_nash) if pi_mono > pi_nash else 0.0
        G.add_node(s, p1=float(prices[0]), p2=float(prices[1]),
                   price_pair=tuple(prices.astype(float)),
                   weight=max(node_gain[s], 1e-6))

    for s in range(M):
        G.add_edge(int(s), int(nxt[s]))

    return {
        'graph': G,
        'next_state': nxt,
        'node_gain': node_gain,
        'pi_nash': pi_nash,
        'pi_mono': pi_mono,
    }


def pick_session_for_phase_diagram(res) -> int:
    """Figure 9 同 Figure 8: 选 1-Sym 中长期价最接近 1.78 的会话."""
    return pick_representative_session(res, prefer_cycle_type=0, prefer_close_to_p98=True)