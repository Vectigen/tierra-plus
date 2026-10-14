# Physical-grid reliability protocol v1

This is a separately trained, synthetic-observation experiment on published
physical atmospheres. It follows the failed hidden-state correction in
[causal-reliability-protocol.md](causal-reliability-protocol.md); that result and
its gates remain unchanged. It does not fit observed transit depths, establish
atmospheric model adequacy, or claim a new inference algorithm.

## Question and prediction

Can nuisance augmentation improve metallicity inference on unseen physical
atmospheres, while retaining useful uncertainty, beyond a matched linear
baseline? A separate classical comparison tests nuisance marginalization and
an interval envelope with abstention. Detecting a problem alone does not count
as corrected inference.

The working prediction is that augmentation reduces paired nuisance response,
but may offer no advantage over augmented ridge or exact finite-grid inference.
Failure of the neural gate ends a claim of special neural benefit for this
setting. The experiment is motivated by objectively scored interventions and
cheap controls, as described in [Nanda's research vision](https://www.neelnanda.io/vision),
not by an affiliation or a claim to have found a neural mechanism.

## Physical assets and frozen split

Use the 192 cloud-free PICASO 3.0 atmospheres in
[Mukherjee et al.'s author archive](https://zenodo.org/records/7236759),
CC-BY-4.0. They vary metallicity, C/O, intrinsic temperature and heat redistribution.
The archive is 31,104,898 bytes, MD5 `4b34c96d22575e6ea65b1f8d3c058092`,
SHA256 `f8515dc3a0cedc91ef78b91ef96264425319fc7bbf1274997468ae8231537c80`.
Individual input hashes and physical coordinates are preserved in the manifest.
The scientific sources are [Mukherjee et al. 2022](https://arxiv.org/abs/2208.07836)
and [Batalha et al. 2019](https://arxiv.org/abs/1904.09355).

The target is **log10 atmospheric metallicity relative to solar**, in dex,
using the grid-coordinate label checked against the stored physical metadata.
This is a physical atmospheric coordinate, not a molecular abundance measurement.

| Split | Intrinsic temperature | Heat redistribution | States |
|---|---|---|---:|
| Training | 100 and 300 K | 0.4 and 0.5 | 128 |
| Calibration | 200 K | 0.4 | 32 |
| Test | 200 K | 0.5 | 32 |

All metallicity/C/O combinations are retained. No physical state crosses splits.
The heldout temperature/redistribution combination is a deliberate generalization
challenge, not an IID random split. The five seeds repeat optimization and noise,
not the selection of 32 new independent atmospheres. Report each physical state's
coverage and error, as well as aggregate metrics.

The PRISM wavelength centers/errors and the fixed lower-bin-edge cut of 2.3 um
are inherited from the previously hashed public inputs. Observed transit depths
are never read. Bin edges are midpoint approximations; piecewise-linear model
spectra are integrated over uniform top-hat bins. No throughput, line-spread
function or error covariance is supplied.

## Interventions and observation model

For physical state θ, wavelength displacement δ, and gray contaminating flux
ratio f, simulate `y_j = bin_j[F_θ(λ + δ)] / (1 + f) + ε_j`.
The Gaussian errors are independent with the published per-bin scales. A free
constant-depth offset is projected out identically for every estimator. Gray
dilution is an explicit constant flux-ratio approximation, not a stellar
atmosphere or spot model.

The shift half-width is one half the median retained bin width, set using
geometry only. Discrete nuisance support is the three shifts `{-halfwidth,0,+halfwidth}`
crossed with `f={0,.05,.10}`. These are stress values, not measured JWST errors.
Primary conditions are clean, gray dilution at .10, both maximum shifts with
zero dilution, and both maximum shifts with .10 dilution. Hold the physical
state and exact noise array fixed across each intervention family.

A secondary condition uses shift `+0.5*halfwidth`, f=.025, outside the *discrete*
reference support but inside its continuous bounding box. It tests discretization;
it is not used for fitting, interval calibration, or selection of gates.

## Estimators, calibration and reference

- A 32-hidden-unit tanh regressor trained only on clean spectra, using the
  existing fixed-budget NumPy implementation: 800 Adam updates, batch 256,
  learning rate 0.003. No early stopping, search or checkpoint selection.
- The same network trained with independent uniform draws from the nine
  nuisance combinations. Its physical states, noise draws and fit seed match
  the clean network. Training has 64 noise draws per physical state.
- Matched ordinary/augmented ridge, training-only standardization, penalty 1.
- Clean networks/ridge use clean-calibration absolute-error intervals. Their
  separate worst-group controls use the largest validation radius across all
  nine nuisance groups. Augmented network/ridge use the same worst-group rule.
  Each group has 32 noise draws per calibration state. The radius is the
  ceil((n+1)*.9)-th absolute-error order statistic. This is a finite-sample
  calibration rule, **not a coverage guarantee under physical-state shift**;
  [conformal coverage assumptions](https://arxiv.org/abs/2107.07511) do not
  automatically hold here.
- Exact finite-grid reference: enumerate all 192 physical states × nine nuisance
  states, with uniform prior mass over each state. Integrate a constant-depth
  offset under a common flat prior; its factor cancels in posterior ratios.
  Sum the Gaussian likelihood and report the posterior mean and central 90%
  metallicity interval. This is exact enumeration of a finite model, not a
  continuous parameter posterior. **The full-grid physics oracle includes the
  generating test states**, unlike the learned estimators.
- Repeat the same reference using only the 128 training states to expose the
  oracle-support advantage. A nominal full-grid reference ignores nuisance.
- The nuisance envelope takes the minimum lower and maximum upper bound of
  the nine nuisance-conditional central 90% intervals. Its point estimate is
  the marginal reference mean. It is conservative within this finite family
  but has no general frequentist guarantee.
- A residual-check control rejects the nominal reference when its minimum
  chi-square exceeds the 95th percentile of clean validation values. This
  separates detection from correction.

All methods use the same abstention threshold: interval width at most **0.6 dex**.
The residual-check control additionally requires its residual check to pass.
Report unconditional coverage/error, interval width, acceptance, and coverage/error
among accepted spectra; an empty accepted set has null coverage and fails.

Calibration-shift detection failures and interval recalibration already appear
in [recent spectral-inference work](https://arxiv.org/html/2606.17098v2).
Neither that failure mode, augmentation, marginalization, nor interval envelopes
are claimed as novel. This repository contributes a reproducible, falsifiable
comparison on explicit physical spectra.

## Frozen gates

Main seeds: 203, 307, 401, 503, 601; 32 noise draws per test state and condition.
All values are fixed in `examples/configs/physical-grid-v1.json` before main.

The augmented-network claim requires **all** of the following in at least 4/5 seeds:

1. In **each** primary condition, unconditional coverage≥.85, acceptance≥.70,
   and accepted coverage≥.85, under the .6dex width cap.
2. Mean primary-condition RMSE≤.90 times augmented-ridge RMSE.
3. Mean paired nuisance-response RMS≤.75 times ordinary-network response.
4. Clean RMSE≤1.10 times ordinary-network clean RMSE + .02dex.

Each classical method separately passes reliability only if criterion 1 holds in
every primary condition, in at least 4/5 seeds. A classical reliability pass is
not a neural-method pass, superiority claim, or proof of continuous-atmosphere
coverage. All failed components and seeds are retained. Secondary/off-grid or
external failure limits the scope even if primary gates pass.

## External mechanism challenge and limits

After the method/config/source are frozen, prepare the separate author
`photochem_cloud_free.zip`: 18,730,827 bytes,
MD5 `74ec83af55eeca6127629f16f01c1409`, same record/license.
These PICASO spectra include VULCAN photochemistry; cite
[Tsai et al. 2021](https://arxiv.org/abs/2108.01790).
Use every supplied state with the same frozen estimators and interventions.
Do not retrain or recalibrate. This is a model-family challenge from related
authors/code/planet, not fully independent observational validation.

This slice can satisfy the *execution* portion of a separately trained physical
grid test; successful reliable-inference promotion still requires the gates.
Finite grid spacing, equal-state priors, the single planet, shared generating
physics, simple instrument response, gray dilution and Gaussian independent
noise prevent broad atmospheric or deployment claims. Fresh radiative transfer,
opacity alternatives, chromatic stellar contamination and real-observation
validation remain outstanding. Existing pRT/Tierra adapter seams remain deferred.

## Reproduction and provenance

Download the two named archives from the author record and verify their listed
checksums. Preparation reads bounded members directly; no archive paths are
extracted. Output directories must be new.

```sh
uv run --with h5py==3.16.0 python -m examples.prepare_physical_grid \
  --archive /path/RCTE_cloud_free.zip --output /path/rcte
uv run --with h5py==3.16.0 python -m examples.prepare_physical_grid \
  --family photochemical --archive /path/photochem_cloud_free.zip --output /path/photochemical
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python -m examples.physical_grid_reliability \
  --grid /path/rcte --external-grid /path/photochemical --output /path/main-v1
```

Main requires clean tracked source and records commit/config/input hashes,
versions, RNG streams, weights, calibration radii, paired draws, all prediction
intervals, per-state metrics and gates. A distinct `--smoke` seed/budget verifies
engineering only and never substitutes for main evidence.
