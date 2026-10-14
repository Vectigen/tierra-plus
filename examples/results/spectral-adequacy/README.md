# Spectral adequacy: the withheld-wavelength check adds little

**Exploratory diagnosis, not fresh validation.** This study reuses every exposed
physical-grid-v1 observation and frozen prediction. Neither estimator is refitted.
All five original seeds and seven nuisance conditions are retained. The method,
controls and analysis rules were committed at `12bbe9d` before this diagnostic run,
but the atmospheric states and original errors were already known.

The candidate predicts each of four fixed wavelength blocks from the other three,
integrating the shared depth offset exactly. It is compared with full-spectrum
chi-square, full-fit block residuals, posterior entropy/width, chemical-family
disagreement, random rejection and uniform interval widening. Read the
[mathematics and analysis rules](../../../docs/spectral-adequacy-diagnosis.md).

**Finding:** withheld-wavelength prediction did not meaningfully improve over
ordinary residuals. At matched acceptance, its small reduction in confidently
wrong predictions falls far short of the prospective 25% target. When thresholds
are instead set using clean calibration spectra, high accepted coverage comes at
the cost of rejecting most photochemical spectra. The useful-acceptance gate fails.

## Matched acceptance on the exposed chemistry shift

Numbers below pool the six primary conditions and five seeds. Each method accepts
`ceil(0.7*n)` observations per condition, or 70.02%. This ranks spectra using the
exposed distribution; it is **not an operational threshold validated on new data**.
An unsafe prediction has absolute point error >0.25 dex and interval width <=0.6 dex.

| Frozen estimator | Selection score | Unsafe fraction among accepted | Accepted nominal 90% coverage |
|---|---|---:|---:|
| Equilibrium grid | Random | 27.94% | 71.59% |
| Equilibrium grid | Full-spectrum chi-square | 9.95% | 90.08% |
| Equilibrium grid | Full-fit block residual | 10.25% | 90.25% |
| Equilibrium grid | Withheld predictive maximum | 9.78% | 90.53% |
| Equilibrium grid | Chemical-family disagreement | 0.15% | 98.51% |
| Augmented network | Random | 23.93% | 66.52% |
| Augmented network | Full-spectrum chi-square | 8.68% | 81.23% |
| Augmented network | Withheld predictive maximum | 8.49% | 81.43% |

The predictive score reduces the grid's pooled unsafe fraction by only about 1.7%
relative to chi-square. All 30 primary seed/condition paired physical-state bootstrap
intervals for that difference include zero. The neural reduction is about 2.2%; its
accepted coverage still misses 85%. These are exploratory comparisons, with no
multiple-comparison correction or fresh-state guarantee.

## Thresholds fixed from clean calibration spectra

Each score's threshold retains at least 90% of its original clean calibration
observations (ties may increase this). It then remains unchanged across conditions.
The same 0.6 dex width cap applies. Values pool the primary conditions and seeds.

| Frozen estimator | Score | In-family acceptance | Shifted acceptance | Shifted accepted coverage |
|---|---|---:|---:|---:|
| Equilibrium grid | Chi-square | 90.88% | 37.67% | 95.58% |
| Equilibrium grid | Withheld predictive maximum | 89.42% | 34.28% | 96.51% |
| Equilibrium grid | Chemical-family disagreement | 93.14% | 96.57% | 73.85% |
| Augmented network | Chi-square | 90.88% | 37.67% | 84.41% |
| Augmented network | Withheld predictive maximum | 89.42% | 34.28% | 84.18% |

The disagreement score's excellent exposed-data ranking does not transfer to a
useful threshold calibrated on equilibrium spectra. It accepts most external
spectra with poor coverage. This distinction prevents a ranking result from being
mistaken for a deployable reliability certificate.

## What including the correct family changes

The union reference gives equal total prior mass to equilibrium and photochemical
families, regardless of their grid sizes. It includes all target physical states
and is therefore an **oracle support control**. On the exposed external challenge,
it accepts 100% and covers 99.99% of discrete labels. Its mean interval width is only
0.0012 dex: this is finite-grid quantization, **not continuous atmospheric precision**.
The envelope of both family-conditional intervals accepts 89.14% with 100% accepted
discrete-label coverage and mean width 0.2146 dex.

Uniform widening calibrated on clean equilibrium observations does not resolve
the chemistry shift: it accepts 99.74% with 71.51% accepted coverage. The results
point to the importance of supplying an adequate physical family. They do not show
that the correct family can be discovered automatically, or that adding these two
families covers clouds, stellar contamination, opacity or all disequilibrium physics.

## Retained artifacts and reproduction

- `report.json.gz`: complete losslessly compressed report, all seeds/conditions,
  selection counts, risks, thresholds and paired-state bootstrap intervals.
- `seed-*.npz`: exact per-observation adequacy scores and oracle predictions.
- `provenance.json` and `record.json`: source, protocol, original report and artifact
  hashes. The five original physical-grid fits/predictions and input grids remain unchanged.

The run took 98.019 seconds on the available Mac, with one BLAS thread and no model
calls or GPU. Its uncompressed report SHA256 is
`7ac74af27583fa0fcd3d16ead61080b951c4d9264f01bc898df5de5bdae22cdf`.

```bash
uv run python -m examples.replay_spectral_adequacy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python -m examples.diagnose_spectral_adequacy --output /path/to/new-output
```

Replay reconstructs every selection from the hashed arrays and independently
checks reported coverage, counts and errors. The inference implementation is tested
against a direct dense correlated-Gaussian mixture, including the offset posterior
covariance, prior weighting, discrete entropy and extreme-misfit normalization.

The next confirmatory test needs fresh physical states and a physical-model
adequacy claim chosen before their outcomes. Repeated noise or reusing these 116
states cannot provide that evidence. The withheld-predictive score should not be
promoted as a special solution on the basis of this diagnosis.
