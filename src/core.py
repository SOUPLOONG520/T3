"""core.py -- Calvano, Calzolari, Denicolo & Pastorello (2020),
"Artificial Intelligence, Algorithmic Pricing, and Collusion", AER 110(10): 3267-3297.

Economic environment, benchmark prices, discrete price grid, payoff tables and
Q-matrix initialisation.  Everything in this module is *closed form or deterministic*
so that it can be unit-checked against the numbers printed in the paper.

Calibration note (important, see docs/CALIBRATION.md)
-----------------------------------------------------
The paper writes the logit demand as

        q_i = exp(mu (a_i - p_i)) / ( sum_j exp(mu (a_j - p_j)) + exp(mu a_0) )

with the baseline parameter "mu = 1/4".  Taken literally that exponent gives a
Bertrand-Nash price-cost margin of ~500% (p^N ~ 6.06) and a monopoly price ~7.19,
i.e. a price grid around [6.0, 7.25].  The paper instead reports a Bertrand margin of
47% ("about twice as large under perfect collusion"), and Table 2 lists deviation /
pre-shock prices on the grid {1.43, 1.51, 1.58, 1.66, 1.74, 1.82, 1.89} and
{1.62, 1.66, 1.70, 1.74, 1.78, 1.82, 1.85, 1.89, 1.93, 1.97}.  Those 17 grid points are
reproduced *exactly* by the exponent coefficient kappa = 1/mu = 4, i.e. by

        q_i = exp((a_i - p_i)/mu) / ( sum_j exp((a_j - p_j)/mu) + exp(a_0/mu) ).

We therefore default to exponent="div" (kappa = 1/mu).  The literal reading is kept
available as exponent="mul" for transparency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "MarketSpec",
    "benchmark_prices",
    "price_grid",
    "payoff_table",
    "init_q_values",
    "encode_profile",
    "profile_index",
    "golden_section_max",
]


# --------------------------------------------------------------------------- #
#  Market
# --------------------------------------------------------------------------- #
@dataclass
class MarketSpec:
    """Stage game.

    n        number of firms (= number of differentiated products)
    quality  a_i, product quality index (vertical differentiation)
    cost     c_i, constant marginal cost
    a0       index of the outside good (inverse index of aggregate demand)
    mu       index of horizontal differentiation / own-price slope
    exponent "div" -> exp((a - p)/mu)   (kappa = 1/mu, reproduces the paper's prices)
             "mul" -> exp(mu (a - p))   (literal printing of eq. (5))
    """

    n: int = 2
    quality: Sequence[float] = (2.0, 2.0)
    cost: Sequence[float] = (1.0, 1.0)
    a0: float = 0.0
    mu: float = 0.25
    exponent: str = "div"

    def __post_init__(self) -> None:
        self.quality = np.asarray(self.quality, dtype=float)
        self.cost = np.asarray(self.cost, dtype=float)
        if len(self.quality) != self.n or len(self.cost) != self.n:
            raise ValueError("quality/cost must have length n")
        if self.mu < 0:
            raise ValueError("mu must be non-negative (mu=0 表示完全替代品)")
        if self.exponent not in ("div", "mul"):
            raise ValueError("exponent must be 'div' or 'mul'")

    # ---- helpers ---------------------------------------------------------- #
    @property
    def kappa(self) -> float:
        """Coefficient multiplying (a - p) inside the exponential."""
        if self.mu <= 0:
            # mu = 0 -> 完全替代品: 份额由"最低价者通吃"给出, 用极大 kappa 逼近
            return 1e6
        return 1.0 / self.mu if self.exponent == "div" else self.mu

    @property
    def a(self) -> np.ndarray:
        return self.quality

    @property
    def c(self) -> np.ndarray:
        return self.cost

    def shares(self, prices: np.ndarray) -> np.ndarray:
        """logit 份额 (softmax 平移后计算, kappa 很大也不会溢出)。"""
        prices = np.asarray(prices, dtype=float)
        logits = self.kappa * (self.quality - prices)
        outside = self.kappa * self.a0
        shift = max(float(logits.max()), outside)
        e = np.exp(logits - shift)
        return e / (e.sum() + np.exp(outside - shift))

    def profits(self, prices: np.ndarray) -> np.ndarray:
        q = self.shares(prices)
        return (np.asarray(prices, dtype=float) - self.cost) * q

    def total_profit(self, prices: np.ndarray) -> float:
        return float(self.profits(prices).sum())

    def quantity(self, prices: np.ndarray, i: int) -> float:
        return float(self.shares(prices)[i])

    def best_response(self, rivals: np.ndarray, i: int,
                      lo: float | None = None, hi: float | None = None) -> float:
        """Firm i's profit-maximising price against fixed rivals' prices."""
        rivals = np.asarray(rivals, dtype=float)
        lo = lo if lo is not None else float(self.cost.min()) - 1.0
        hi = hi if hi is not None else float(self.quality.max()) + 25.0 / self.kappa

        def obj(p: float) -> float:
            return float(self.profits(np.insert(rivals, i, p))[i])

        return unimodal_max(obj, lo, hi)

    def with_prices(self, prices: Sequence[float]) -> "MarketSpec":
        """A copy of the market with firm 1's quality rescaled (helper for tests)."""
        return MarketSpec(n=self.n, quality=self.quality, cost=self.cost,
                          a0=self.a0, mu=self.mu, exponent=self.exponent)


# --------------------------------------------------------------------------- #
#  Numerical helpers
# --------------------------------------------------------------------------- #
def golden_section_max(f, lo: float, hi: float, tol: float = 1e-10,
                       max_iter: int = 300) -> float:
    """Derivative-free maximisation of a unimodal f on [lo, hi]."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def unimodal_max(f, lo: float, hi: float, n_grid: int = 240,
                 tol: float = 1e-10) -> float:
    """Robust maximisation: coarse grid to locate the basin, then golden section.

    Profits underflow to exactly 0 far away from the optimum, which makes a pure
    golden-section search drift to the boundary; the grid stage avoids that.
    """
    xs = np.linspace(lo, hi, n_grid)
    fs = np.array([f(x) for x in xs])
    j = int(np.argmax(fs))
    a = xs[max(j - 1, 0)]
    b = xs[min(j + 1, n_grid - 1)]
    if b - a < tol:
        return float(xs[j])
    return golden_section_max(f, float(a), float(b), tol=tol)


def _coordinate_max(func, x0: np.ndarray, lo: float, hi: float,
                    sweeps: int = 60, tol: float = 1e-10) -> np.ndarray:
    """Coordinate-wise golden-section maximisation (used for asymmetric maxima)."""
    x = np.array(x0, dtype=float)

    def make_f(j: int):
        def f(v: float) -> float:
            y = x.copy()
            y[j] = v
            return func(y)
        return f

    for _ in range(sweeps):
        improved = False
        for j in range(len(x)):
            v = unimodal_max(make_f(j), lo, hi)
            if abs(v - x[j]) > 1e-12:
                improved = True
            x[j] = v
        if not improved:
            break
    return x


def benchmark_prices(spec: MarketSpec) -> dict:
    """Bertrand-Nash static prices p^N and joint-profit-maximising prices p^M.

    p^N is found by iterating firm-wise best responses from several starts
    (profit is unimodal in own price for logit demand); p^M by coordinate search.
    """
    # ---- p^N ------------------------------------------------------------- #
    starts = [np.full(spec.n, float(spec.quality[0]))]
    starts.append(spec.quality.astype(float).copy())
    starts.append(np.maximum(spec.quality - 0.5 * (spec.quality - spec.cost), spec.cost))
    best = None
    for x0 in starts:
        x = np.array(x0, dtype=float)
        for _ in range(200):
            x_new = np.array([spec.best_response(np.delete(x, i), i) for i in range(spec.n)])
            if np.max(np.abs(x_new - x)) < 1e-12:
                x = x_new
                break
            x = 0.5 * (x + x_new)
        # keep only solutions that are mutual best responses
        ok = all(abs(spec.best_response(np.delete(x, i), i) - x[i]) < 1e-6 for i in range(spec.n))
        if ok and (best is None or spec.total_profit(x) < spec.total_profit(best)):
            best = x
    p_nash = np.array(best if best is not None else starts[0], dtype=float)

    # ---- p^M ------------------------------------------------------------- #
    p_mono = _coordinate_max(
        spec.total_profit,
        np.maximum(p_nash + 0.1, spec.cost),
        lo=float(spec.cost.min()) - 1.0,
        hi=float(spec.quality.max()) + 25.0 / spec.kappa,
    )

    pi_nash = spec.profits(p_nash)
    pi_mono = spec.profits(p_mono)
    return {
        "p_nash": p_nash,
        "p_mono": p_mono,
        "pi_nash": pi_nash,               # per-firm profit in static Bertrand-Nash
        "pi_nash_avg": float(pi_nash.mean()),
        "pi_mono": pi_mono,               # per-firm profit under full collusion
        "pi_mono_avg": float(pi_mono.mean()),
        "q_nash": spec.shares(p_nash),
        "q_mono": spec.shares(p_mono),
        "margin_nash": float(np.mean((p_nash - spec.cost) / spec.cost)),
        "margin_mono": float(np.mean((p_mono - spec.cost) / spec.cost)),
    }


def price_grid(p_nash: float, p_mono: float, m: int = 15, xi: float = 0.1,
               mode: str = "sym") -> np.ndarray:
    """m equally spaced prices in [p^N - xi(p^M-p^N), p^M + xi(p^M-p^N)].

    mode "sym"   : as in the paper.
    mode "down"  : grid enlarged only downwards (robustness check, Section V.G).
    mode "wide"  : larger xi (robustness check).
    """
    span = p_mono - p_nash
    if mode == "sym":
        lo, hi = p_nash - xi * span, p_mono + xi * span
    elif mode == "down":
        lo, hi = p_nash - xi * span, p_mono
    elif mode == "wide":
        lo, hi = p_nash - 3 * xi * span, p_mono + 3 * xi * span
    else:
        raise ValueError(mode)
    return np.linspace(lo, hi, m)


# --------------------------------------------------------------------------- #
#  Discrete payoffs on the action grid
# --------------------------------------------------------------------------- #
def profile_index(actions: Sequence[int] | np.ndarray, m: int) -> int:
    """Base-m encoding of an action profile (row index of the payoff table)."""
    idx = 0
    for a in actions:
        idx = idx * m + int(a)
    return idx


def encode_profile(actions: np.ndarray, m: int) -> np.ndarray:
    """Vectorised base-m encoding; actions has shape (..., n) -> int array (...)."""
    out = np.zeros(actions.shape[:-1], dtype=np.int64)
    for j in range(actions.shape[-1]):
        out = out * m + actions[..., j].astype(np.int64)
    return out


def payoff_table(spec: MarketSpec, grid: np.ndarray) -> np.ndarray:
    """pi_table[r, i] = profit of firm i under action profile r.

    Shape (m**n, n).  Row r enumerates all action profiles in base-m order with the
    *slowest* varying index being firm n (matches profile_index/encode_profile).
    """
    m = len(grid)
    profiles = np.array(list(itertools.product(range(m), repeat=spec.n)), dtype=np.int64)
    prices = grid[profiles]                       # (m**n, n)
    profits = np.empty_like(prices, dtype=float)
    for r in range(profits.shape[0]):
        profits[r] = spec.profits(prices[r])
    return profits


def rival_average_profit(spec: MarketSpec, pi_table: np.ndarray, m: int) -> np.ndarray:
    """pi_bar[i, a] = average profit of firm i when it plays a and rivals uniformise.

    This is the numerator of the initial Q-matrix, eq. (8).
    """
    n = spec.n
    out = np.zeros((n, m), dtype=float)
    idx = np.arange(pi_table.shape[0])
    for i in range(n):
        digits = (idx // (m ** (n - 1 - i))) % m          # i-th action of every profile
        for a in range(m):
            out[i, a] = pi_table[digits == a, i].mean()
    return out


def init_q_values(spec: MarketSpec, pi_table: np.ndarray, m: int,
                  delta: float) -> np.ndarray:
    """Q_{i,0}(s, a) = pi_bar^i(a) / (1 - delta)   (eq. 8, independent of the state).

    Returns shape (n, m): one row per firm, the same for every state.
    """
    pi_bar = rival_average_profit(spec, pi_table, m)      # (n, m)
    return pi_bar / max(1.0 - delta, 1e-6)