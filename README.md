# Replication: Artificial Intelligence, Algorithmic Pricing, and Collusion

An **independent, third-party replication** of

> Emilio Calvano, Giacomo Calzolari, Vincenzo Denicolò, Sergio Pastorello.
> *"Artificial Intelligence, Algorithmic Pricing, and Collusion."*
> **American Economic Review** 110(10): 3267–3297. DOI: [10.1257/aer.20190623](https://doi.org/10.1257/aer.20190623)

Written from the published paper alone. This repository is **not** affiliated with,
endorsed by, or derived from the authors' own code or their ICPSR replication package
([10.3886/E119462V1](https://doi.org/10.3886/E119462V1)).

---

## TL;DR — what this replication finds

The paper's central claim reproduces: **Q-learning pricing agents with no communication,
no instruction to collude, and no prior knowledge of the market learn to charge
supracompetitive prices, and those prices are supported by an equilibrium
reward–punishment structure rather than by failure to optimise.**

| Result | Paper | This replication | |
|---|---|---|---|
| Average profit gain Δ (α=0.15, β=4e-6, 300 sessions) | 0.849 | **0.8368** | ✅ |
| Modal steady-state price | 1.78 (≈19%) | **1.78 (16.0%)**, never reaches the monopoly price | ✅ |
| Cycle mix (const-sym / const-asym / period-2 / ≥3) | 27.7 / 36.6 / 23.8 / 11.9 % | **25.0 / 35.7 / 23.0 / 16.3 %** | ✅ |
| Q-loss, all states (Table 1) | 0.018 | **0.0159** | ✅ |
| Table 3B: frequency of unprofitable deviations | > 0.95 | **0.951** | ✅ |
| Table 2A: non-deviator's price cut at τ=2 (pre-shock 1.78) | ≈ −13% | **−12.53%** | ✅ |
| Strategy graph: sessions where every node returns to the longrun price | 0.92 | **0.920** | ✅ |
| Strategy graph: sessions with <3 non-convergent nodes | 0.98 | **0.983** | ✅ |
| Strategy graph: path length mean / max | 6 / 18 | **5.23 / 17** | ✅ |
| Figure 11 (rematching converged agents): Δ falls to ≈0.20 | — | **0.85 → 0.20, recovers in <5,000 periods** | ✅ |
| corr(Δ, punishment intensity) | +0.762 | **+0.558** | ◐ |
| Frequency of on-path Nash play | 0.505 | **0.000** | ❌ [diagnosed](#the-one-quantity-we-do-not-reproduce) |

Scale differences (300 vs 1,000 sessions; 20×20 vs 100×100 grid; 3e6 vs 1e9 period cap)
are listed in [`docs/REPLICATION_REPORT.md`](docs/REPLICATION_REPORT.md) §7.

---

## ⚠️ A calibration inconsistency in the paper — please read before re-implementing

The paper prints the demand function as `exp(mu (a_i - p_i))` with baseline `mu = 1/4`.
Implemented **literally**, that yields a Bertrand price–cost margin of ≈500%, which
contradicts the paper's own statement that the margin is "approximately 47 percent"
(and "about twice as large" under full collusion), and contradicts the price grid printed
in its Table 2.

Everything lines up exactly if the exponent coefficient is `kappa = 1/mu = 4`, i.e. if
demand is `exp((a_i - p_i)/mu)`. Under `kappa = 4` three independent checks agree:

* margins: `p^N = 1.47293` → **0.4729**; `p^M = 1.92498` → **0.9250**;
* the 15-point grid is `1.4277 … 1.9702`, containing **all 17 prices printed in Table 2**;
* the paper's footnote 21 normalisation (Δ ≈ −2% "by defect", ≈ 12% "by excess")
  reproduces as **−2.38% / +11.66%**.

Full derivation and arguments: **[`docs/CALIBRATION.md`](docs/CALIBRATION.md)**.
Reproduce the check yourself with `python tests/test_core.py`.

---

## What is implemented

* **Environment** — logit demand (eq. 5), benchmark prices `p^N`/`p^M`, the 15-point
  discrete price grid spanning below Bertrand to above monopoly (§II.B), bounded memory
  `s_t = {p_{t-1},…,p_{t-k}}` (eq. 6), initial matrix `Q_0` (eq. 8).
* **Algorithm** — ε-greedy Q-learning with time-declining exploration `ε_t = exp(-βt)`
  (eq. 7) and the update rule (eq. 4); low-price tie-breaking; convergence criterion =
  arg-max map unchanged for 100,000 consecutive periods (§III.B).
* **Diagnostics** — the paper's ex-post mechanism test: given the rival's limit strategy,
  solve the "true Q" system `Q(s,a) = π(s,a) + δ·max_{a'} Q(T(s,a),a')` by value iteration,
  then compare arg-maxes to obtain Nash frequency and Q-loss (§III.D).
* **Strategy analysis** — forced-deviation impulse responses (Tables 2–3, Figures 4–6)
  and the directed graph of limit strategies (Figures 8–9).
* **Robustness** — δ sweep, number of firms, cost asymmetry, stochastic demand,
  differentiation μ, initialisation, action-set size; learning curve and rematching
  (Figures 10–11).

---

## Repository layout

```
README.md                      this file
CITATION.cff                   how to cite (neutral authorship, see "Provenance")
LICENSE                        MIT
requirements.txt               numpy, matplotlib, networkx
MANIFEST.sha256                sha256 of every shipped file (self-excluded)

docs/
  CALIBRATION.md               the kappa = 1/mu finding (full derivation)
  REPLICATION_REPORT.md        item-by-item measured vs published numbers (Chinese)
  SANDBOX_NOTES.md             machine-specific notes (not needed elsewhere)
  COMMON.md                    internal development spec

src/
  core.py                      environment: demand, benchmark prices, price grid, payoffs, Q_0
  engine.py                    vectorised Q-learning engine (slot-flattened, session eviction)
  analysis.py                  true-Q value iteration, best response, Q-loss, deviation gain
  parallel.py                  dispatch helpers
  deviation.py                 §IV.B deviation/punishment analysis
  strategy_graph.py            §IV.C strategy graphs
  robustness.py                §V robustness experiments
  timescale.py                 §VI learning curve and rematching

run_representative.py          representative experiment → results/representative.npz
run_grid.py                    (α,β) grid → Figures 1/2/7  (supports --shard/--nshard/--merge)
run_table1.py                  Table 1-style statistics from the representative run
run_deviation.py               Tables 2–3, Figures 4–6
run_strategy_graph.py          Figures 8–9
run_robustness.py              Figure 3, Table 4, number of firms, demand/μ/init sweeps
run_timescale.py               Figures 10–11
run_nash_long.py               diagnostic for the on-path Nash frequency

tests/
  test_core.py                 calibration check (three independent hits)
  test_engine_ref.py           vectorised engine vs a naive per-session reference
  test_analysis.py             vectorised vs scalar best-response/Q-loss evaluation
  test_compaction.py           session-eviction regression
  bench_loop.py, profile_run.py, diag_traj.py, smoke_engine.py   profiling helpers

results/                       CSVs, JSONs and .npz produced by the runs
figures/                       Figures 1–11
```

---

## How to run

```bash
pip install -r requirements.txt

# 1. verify the implementation (fast, ~1 min)
python tests/test_core.py
python tests/test_analysis.py
python tests/test_compaction.py
python tests/test_engine_ref.py 12000 12 500      # slow: naive reference

# 2. representative experiment (300 sessions × up to 3e6 periods; ~17 min single process)
REP_SESSIONS=300 REP_MAX_PERIODS=3000000 REP_PATIENCE=100000 python run_representative.py
python run_table1.py

# 3. deviation analysis and strategy graphs (read the .npz from step 2)
python run_deviation.py
python run_strategy_graph.py

# 4. (α,β) grid → Figures 1/2/7.  Shard it across cores, then merge:
for k in 0 1 2 3 4 5; do
  python run_grid.py --nx 20 --ny 20 --sessions 3 --periods 2500000 \
                     --tag main --shard $k --nshard 6 &
done; wait
python run_grid.py --nx 20 --ny 20 --tag main --merge

# 5. robustness and time scale
python run_robustness.py
python run_timescale.py
python run_nash_long.py --sessions 8 --periods 1000000 3000000 8000000
```

**A single session needs 2–4 million periods to converge** (with β=4e-6, ε only
reaches 1e-3 after ≈690,000 periods). Budget accordingly.

---

## The one quantity we do not reproduce

The paper reports an **on-path Nash frequency of 0.505**; we obtain 0.000. This is not
evidence that the agents fail to optimise — the Q-loss magnitudes match closely
(0.0068 on path, 0.0159 across all states, vs the paper's 0.001 / 0.018).

We tested the obvious explanation, "not converged long enough", by running to 8,000,000
periods (ε ≈ 1e−14). The results **stop changing** between 3e6 and 8e6 periods, so
convergence is not the cause. Our best explanation: Q-learning updates only visited cells,
so once ε → 0 the **off-path cells freeze at stale values**, while our value iteration uses
the true Q; the two arg-maxes then flip between adjacent grid points whose Q values are
nearly tied. The paper's higher frequency may reflect its much slower exploration decay,
which keeps off-path cells refreshed. Details and the diagnostic table:
[`docs/REPLICATION_REPORT.md`](docs/REPLICATION_REPORT.md) §5.

---

## Provenance

This repository was produced as **coursework, with AI assistance** (Claude/DeepSeek
family models) driving the implementation, the numerical experiments and the writing.
Authorship is therefore recorded neutrally in `CITATION.cff` rather than attributed to an
individual.

What *was* done by hand and is verifiable from this repository alone:

* the reading of the paper and the specification of every equation implemented;
* the identification of the calibration inconsistency (§ above) and its derivation;
* every design decision recorded in the code comments and in `docs/REPLICATION_REPORT.md`;
* the checks in `tests/`, which fail loudly if the implementation is wrong.

Anyone using this for anything beyond coursework should re-derive the calibration claim
(`docs/CALIBRATION.md`) and re-run the verification scripts before relying on the results.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{calvano2020ai,
  author  = {Calvano, Emilio and Calzolari, Giacomo and Denicol\`o, Vincenzo
             and Pastorello, Sergio},
  title   = {Artificial Intelligence, Algorithmic Pricing, and Collusion},
  journal = {American Economic Review},
  volume  = {110}, number = {10}, pages = {3267--3297}, year = {2020},
  doi     = {10.1257/aer.20190623}
}
```

## Notes on scope

* The paper PDF is **not** redistributed here (AER copyright). See the DOI above.
* The code was written by reading the paper; the authors' own replication package was
  not consulted.
* Figures 3, 10, 11 and Table 4 were produced under a reduced-compute protocol
  (fixed 0.5–0.8M periods rather than convergence), so their **levels** are lower than
  the paper's; compare trends, not magnitudes. See the report.

## License

MIT — see [`LICENSE`](LICENSE).