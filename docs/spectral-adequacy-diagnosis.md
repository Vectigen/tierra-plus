# Withheld-wavelength adequacy: exploratory diagnosis

This analysis uses only the already exposed physical-grid-v1 observations and
frozen predictions. It is **not a new heldout experiment**. The 192 equilibrium
states, 116 photochemical states, noise draws and failed chemistry-shift results
have already been inspected. No result here can clear the prospective fresh-grid
generalization gate. The original experiment and its negative results remain
unchanged.

The question is specific: can inconsistency between wavelength blocks distinguish
confident metallicity errors better than an ordinary full-spectrum residual? The
first analysis changes abstention, not the estimator. It evaluates both frozen
finite-grid and augmented neural predictions; no model is refitted.

## Fixed calculation

Partition the 127 retained bins into four contiguous groups with `array_split`
in ascending wavelength order. No band is chosen from an observed error. For
each withheld block B, fit the finite equilibrium atmosphere/nuisance grid using
only its complement A. Keep the original independent Gaussian error model and
common flat prior for a constant transit-depth offset.

For candidate j, the offset posterior has mean
`c_j = sum_A((y-m_j)/sigma²) / sum_A(1/sigma²)` and variance
`v = 1 / sum_A(1/sigma²)`. The withheld predictive distribution is the mixture
of Gaussians with mean `m_j,B+c_j` and covariance
`diag(sigma_B²) + v*11ᵀ`, weighted by the A-only candidate posterior. Thus the
withheld bins do not enter their own fit, and shared-offset uncertainty is not
discarded.

The implementation uses the equivalent exact evidence ratio. If
`Z_S = sum_j prior_j exp(-chi²_j,S/2)` and `P_S=sum_S(1/sigma²)`, the score is
`[2(log Z_A-log Z_full)+log(P_full/P_A)] / number_of_withheld_bins`.
This is minus twice the predictive log density after subtracting the known
independent-noise Gaussian normalizer, per withheld bin. The candidate score is
the maximum over all four blocks. A threshold is calibrated on that maximum
once. This is not four separate uncorrected hypothesis tests. Dense correlated
Gaussian mixture fixtures independently test the identity.

## Controls and selection

All adequacy scores are computed from the fixed equilibrium reference:

- full-spectrum minimum chi-square;
- maximum block residual at the full-spectrum best candidate and fitted offset;
- posterior metallicity entropy and central interval width;
- chemical-family conditional mean disagreement;
- random rejection and always accept.

The chemical comparison explicitly adds the exposed photochemical family as an
**oracle support control**. The joint union has half of its prior mass in each
family, independent of the number of grid rows. We also report the envelope of
the two family-conditional central intervals, centered on the union posterior
mean. Both supports include the target physical states. Neither result establishes
reliability for omitted chemistry, clouds, opacity, stars or continuous parameters.

Two selection summaries are fixed before running this diagnosis:

1. The 90th percentile (`higher`) of each score on the original clean validation
   observations, independently for each of the five original seeds. Apply the
   resulting threshold to every condition, also enforcing the original 0.6 dex
   interval width cap. Ties may make calibration acceptance exceed 90%.
2. Within each reported family/condition, accept the lowest-scoring `ceil(0.7*n)`
   observations. Break score ties with fixed random values independent of labels.
   This is a matched-acceptance **exploratory ranking comparison**, not an
   operational threshold learned from untouched validation. The random control
   has exactly the same accepted count. Always accept is reported separately.

The error event is absolute point error >0.25 dex with interval width <=0.6 dex.
Report unsafe accepted fraction, accepted coverage, accepted RMSE and acceptance.
Coverage is coverage of the discrete label, never a continuous uncertainty claim.
The paired difference between predictive and chi-square selection uses 2,000
physical-state bootstrap replicates per family/condition/seed. All repeated draws
of a physical state stay together. These are exploratory percentile intervals,
not simultaneous confidence statements across all comparisons.

A uniform widening control uses the conservative 90% absolute-error calibration
radius of the equilibrium predictor on the same clean validation set, adding
`mean±radius` to its existing interval. We retain its actual acceptance under the
same width cap. It is not tuned to chemistry-shift outcomes.

All seven original conditions, both evaluated families and all five original
seeds are retained. Original predictions are checksum verified. Recomputed grid
predictions must match saved values. Noise is regenerated from the original
streams; no additional noise realizations are represented as new physical data.

## Interpretation and next gate

An apparent score association on this diagnosis is a hypothesis for a new study.
Reject the special withheld-predictive claim if cheap controls match it or its
coverage–acceptance tradeoff is poor. A detector that rejects almost everything
does not supply reliable inference. Strong oracle-union performance would show
the value of including the right physical family, not automatic discovery of it.

A confirmatory study still requires a separately acquired physical family and
physical-state calibration/test split fixed before outcomes, with at least 32
fresh test states. Thresholds may use only calibration states. The prospective
gate is >=85% accepted coverage and >=70% acceptance in each condition in >=4/5
seeds; at matched70% acceptance, at least25% fewer unsafe accepted predictions than
chi-square with a paired physical-state interval excluding no improvement; clean
acceptance>=85%; and accepted RMSE no worse than chi-square+0.02 dex. This diagnosis
cannot meet that gate regardless of its numerical results.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python -m examples.diagnose_spectral_adequacy --output /path/to/new-output
```

Run from a clean committed source checkout. Output must be new and outside the
checkout. Only NumPy and the existing prepared public grids are required. The
runner writes source/input provenance, exact per-observation score archives,
selection summaries and paired-state bootstrap results; it retains a failure
record if a run stops. The earlier [physical-grid protocol](physical-grid-protocol.md)
and [data attribution](../examples/data/physical-grid/README.md) continue to apply.
No new radiative transfer, real observed-depth retrieval or external GPU is used.
