"""parallel.py -- 进程池调度。

numpy 内层循环的每周期固定开销较大, 因此:
  * 同一个环境下的多个实验点 (alpha, beta) 会被合并成一批 (run_batch), 提高吞吐;
  * 每批再按会话数切片分给多个进程;
  * 每个进程保证至少 ~200 行, 以摊薄 numpy 调用开销。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np

from engine import SessionResult, SimConfig, build_env, run_batch, run_chunk

DEFAULT_WORKERS = min(20, (os.cpu_count() or 4))


def _run_one(payload):
    """进程池入口 (Windows 用 spawn, 必须是模块级函数)。"""
    cfg, sessions, seed, env, return_q = payload
    return run_batch([cfg], int(sessions), seed=int(seed), env=env, return_q=return_q)


def _run_points(payload):
    cfgs, sessions, seed, env, return_q = payload
    return run_batch(list(cfgs), int(sessions), seed=int(seed), env=env, return_q=return_q)


def split_chunks(total: int, n_workers: int, min_rows: int = 200) -> list[tuple[int, int]]:
    """把 total 行切成若干块, 每块至少 min_rows 行 (除非总量太小)。"""
    if total <= 0:
        return []
    n_workers = max(1, min(n_workers, max(1, total // min_rows)))
    size = int(np.ceil(total / n_workers))
    out = []
    start = 0
    while start < total:
        out.append((start, min(start + size, total)))
        start += size
    return out


def run_config(cfg: SimConfig, n_sessions: int, n_workers: int = DEFAULT_WORKERS,
               base_seed: int = 0, min_rows: int = 200,
               return_q: bool = False) -> SessionResult:
    """单个实验点, 会话维度并行。"""
    env = build_env(cfg)
    chunks = split_chunks(n_sessions, n_workers, min_rows)
    payloads = [(cfg, hi - lo, base_seed + 1000 * k, env, return_q)
                for k, (lo, hi) in enumerate(chunks)]
    parts = _dispatch(_run_one, payloads, n_workers)
    return _concat(parts)


def run_configs(cfgs: Sequence[SimConfig], n_sessions: int,
                n_workers: int = DEFAULT_WORKERS, base_seed: int = 0,
                min_rows: int = 200, return_q: bool = False) -> list[SessionResult]:
    """同环境的多个实验点: 合并成批后并行, 返回与 cfgs 等长的结果列表。"""
    cfgs = list(cfgs)
    env = build_env(cfgs[0])
    # 估计总行数 -> 需要多少进程
    total_rows = len(cfgs) * n_sessions
    n_use = max(1, min(n_workers, max(1, total_rows // min_rows)))
    size = int(np.ceil(len(cfgs) / n_use))
    payloads, spans = [], []
    for k, start in enumerate(range(0, len(cfgs), size)):
        sub = cfgs[start:start + size]
        payloads.append((sub, n_sessions, base_seed + 1000 * k, env, return_q))
        spans.append((start, start + len(sub)))
    parts = _dispatch(_run_points, payloads, n_workers)
    out: list[SessionResult] = [None] * len(cfgs)          # type: ignore[list-item]
    for (lo, hi), res in zip(spans, parts):
        n_sess = res.meta["sessions"]
        for p in range(hi - lo):
            rows = np.arange(p * n_sess, (p + 1) * n_sess)
            out[lo + p] = res.subset(rows)
    return out


def _dispatch(fn, payloads, n_workers):
    if len(payloads) == 1:
        return [fn(payloads[0])]
    with ProcessPoolExecutor(max_workers=min(n_workers, len(payloads))) as ex:
        return list(ex.map(fn, payloads))


def _concat(parts: Sequence[SessionResult]) -> SessionResult:
    from engine import merge
    return merge(list(parts))