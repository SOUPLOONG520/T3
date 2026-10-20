# A calibration inconsistency in the paper — and how we resolved it

This note documents the single most consequential finding of our replication.
Anyone re-implementing this paper from the printed text alone will not be able
to reproduce its reported numbers unless they notice it.

## 1. What the paper prints

Section II.A, eq. (5), gives logit demand with a quality index `a_i`, an outside
good with index `a_0`, and a horizontal-differentiation index `mu`:

```
q_i = exp(mu (a_i - p_i)) / ( sum_j exp(mu (a_j - p_j)) + exp(mu a_0) )
```

Section II.E gives the baseline parameters:

```
n = 2,  c_i = 1,  a_i - c_i = 1 (hence a_i = 2),  a_0 = 0,  mu = 1/4,
delta = 0.95,  m = 15,  xi = 0.1,  k = 1
```

## 2. What breaks if you implement that literally

Solving the symmetric Bertrand–Nash equilibrium and the joint-profit maximum
under `mu = 1/4` **in the exponent as printed** gives

| quantity | literal reading | paper's own statements |
|---|---|---|
| price–cost margin at Bertrand–Nash | ≈ 500% (p^N ≈ 6.06) | "approximately 47 percent" |
| monopoly price | ≈ 7.19 (margin ≈ 619%) | "about twice as large" |
| price grid (15 points) | ≈ [6.0, 7.25] | Table 2 lists prices in [1.43, 1.97] |

So the literal reading is inconsistent with the paper's prose *and* with its
own Table 2. This is not a rounding issue.

## 3. The resolution

The numbers are reproduced exactly if the exponent coefficient is
`kappa = 1/mu = 4`, i.e. if demand is

```
q_i = exp((a_i - p_i)/mu) / ( sum_j exp((a_j - p_j)/mu) + exp(a_0/mu) )
```

Under `kappa = 4` three *independent* checks line up:

**(a) Margins.** `p^N = 1.47293` → margin `0.4729`; `p^M = 1.92498` → margin
`0.9250`, i.e. "approximately 47 percent" and "about twice as large".

**(b) The price grid.** With `m = 15`, `xi = 0.1`, prices are the 15 equally
spaced points of `[p^N − 0.1(p^M − p^N), p^M + 0.1(p^M − p^N)]`:

```
1.4277 1.4665 1.5052 1.5440 1.5827 1.6215 1.6602 1.6990 1.7377 1.7764
1.8152 1.8539 1.8927 1.9314 1.9702
```

Rounded to two decimals these are `1.43 … 1.97` in steps of 0.04, and they
contain **every one of the 17 prices printed in Table 2** — the deviation
prices `1.43, 1.51, 1.58, 1.66, 1.74, 1.82, 1.89` (the paper prints every
second deviation price) and the pre-shock prices `1.62 … 1.97`.

**(c) The two normalisations quoted in footnote 21.** The paper states that
Δ ≈ −2% if the Nash price is approximated *by defect* and Δ ≈ 12% if it is
approximated *by excess*. Evaluating eq. (9) at the symmetric grid points just
around `p^N` gives −2.38% and +11.66%.

We therefore implement `kappa = 1/mu` as the default (`MarketSpec(exponent="div")`)
and keep the literal reading available as `exponent="mul"` for comparison.

## 4. How to check this yourself

```bash
python tests/test_core.py
```

It prints `p^N`, `p^M`, the margins, the 15-point grid, and explicitly tests
that all 17 Table-2 prices are grid points.

## 5. Why we think this is the right reading

Three arguments, in decreasing order of confidence:

1. **Numerical**: (a)–(c) above are three independent coincidences; the
   probability that the wrong exponent reproduces all three is negligible.
2. **Internal**: footnote 26 ("with our discretisation of the price space, the
   average profit gain would already be close to 12 percent if the Nash prices
   were just approximated by excess") only makes sense on the `kappa = 4` grid.
3. **Typographic**: the paper's LaTeX renders `exp` of a fraction with the
   denominator `mu`; a PDF text extraction shows `mu` in a superscript position,
   which is exactly how `\frac{a-p}{\mu}` can be mis-read as `mu (a-p)`.

**This is a statement about the printed text, not about the authors' code.** We
did not consult the original replication package (DOI 10.3886/E119462V1) and
derived this independently from the paper's own reported numbers.

## 6. Practical advice for replicators

If your implementation gives price–cost margins in the hundreds of percent, or
a price grid nowhere near [1.43, 1.97], your run is not wrong — you have simply
taken eq. (5) literally. Set `kappa = 1/mu` and re-check against Table 2 before
interpreting anything else.