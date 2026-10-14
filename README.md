# Tierra+

**Causal reliability of spectral inverse models.**

Tierra+ tests when spectral estimates become confident and wrong after the data
or physical model changes. Its experiments use paired nuisance interventions,
held-out atmospheric states, and exact reference calculations to distinguish
useful corrections from reassuring diagnostics. Both successful controls and
failed hypotheses are retained.

## Run a reliability audit

The supported `audit` command goes from a supplied finite spectral grid to one
inference with an empirical accept/abstain decision and explicit failure reasons.
This copy-paste demonstration uses the bundled, already exposed physical assets:

```bash
git clone https://github.com/Vectigen/tierra-plus.git
cd tierra-plus
uv sync --dev
AUDIT_DEMO=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m examples.reliability_audit_demo --output "$AUDIT_DEMO/run"
```

It creates four recorded cases: full-state qualification, an outcome-informed
interior-label software example, a fixed photochemical observation, and a geometry
rejection. Read the [standalone result](examples/results/reliability-audit/README.md)
and the [bundle input format](docs/reliability-audit.md#input-bundles) before using
your own support, calibration, validation and observation data. A
`conditional_accept` remains conditional on the supplied finite model family and
empirical validation; it is not a general reliability certificate.

The bundled physical demo's lead full panel uses all 32 exposed RCTE calibration
states and all 32 exposed RCTE validation states. Under the corrected operational
rule it abstains: 22/32 states are accepted (68.75%, below the 70% floor), while all
22 accepted intervals cover their discrete labels. A separately labelled,
outcome-informed interior-label panel accepts 11/12 states and exists only to
exercise the accept path; it is invalid as general RCTE qualification. An earlier
demo that omitted boundary abstention from validation is retained as invalidated
development evidence, not as a research acceptance.

The first controlled experiment uses **synthetic observations and known synthetic labels**, with its
grid, error scale and target shape anchored to hashed public WASP-39b PRISM assets.
The target is a template amplitude, not a molecular abundance. This controlled
problem makes paired interventions and exact reference solutions available; it
does not establish a failure in a deployed atmospheric retrieval.

For the first controlled study, read the [frozen protocol](docs/causal-reliability-protocol.md),
[experiment configuration](examples/configs/causal-reliability-v1.json), and
[experiment records](examples/results/causal-reliability/README.md).
The protocol fixes the hypotheses, controls, splits and decision rules before the
main experiment. A hypothesis may fail; running the benchmark is not evidence that
its correction works.

**First result:** all five primary runs reproduced the specified prior-shift
failure: mean nominal 90% coverage fell from 89.6% to 45.3%. The proposed hidden
correction passed zero of five runs and was dominated by the known-geometry GLS
control. Exact training-prior Bayes also failed under the changed prior, while the
network's intervention effects exceeded the predeclared oracle-explanation
tolerance. Read the [results and retained negative findings](examples/results/causal-reliability/README.md).

**Physical-grid result:** a separately trained nuisance-augmented network passed
the predefined gate in 4/5 seeds on held-out PICASO equilibrium atmospheres,
improving on matched ridge. A training-support physical-grid reference was still
more accurate. On 116 external photochemical atmospheres, the neural estimator's
nominal 90% coverage fell to 66.5% despite accepting every spectrum; the grid
reference also became overconfident. Its often zero-width intervals describe
discrete metallicity labels, not continuous atmospheric precision.

Read the [physical protocol](docs/physical-grid-protocol.md) and
[complete results, width/abstention tradeoffs and reproduction commands](examples/results/physical-grid/README.md).
The subsequent [adequacy diagnosis](examples/results/spectral-adequacy/README.md)
found that withholding wavelength blocks adds little over ordinary residuals.
High accepted coverage required rejecting most shifted spectra. Including the
correct chemical family helped an oracle reference, but this does not identify
missing physics automatically. These exposed states remain diagnosis data;
a new reliability claim requires fresh held-out physics.

## Reproduce the causal experiment

```bash
git clone https://github.com/Vectigen/tierra-plus.git
cd tierra-plus
uv sync --dev
uv run pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --with h5py==3.16.0 python -m examples.causal_reliability \
  --config examples/configs/causal-reliability-v1.json \
  --output /tmp/tierra-causal
```

The predictor and analytic controls use NumPy. HDF5 loading requires `h5py`; plotting
is separate and optional. No GPU, hosted model or paid service is required.

## What would change the conclusion?

**Prior shift:** a small neural predictor must first fit the training distribution
well. Only then can poor prediction and interval coverage under a reversed nuisance
correlation qualify as the specified failure. The exact training-prior Bayes
estimator is a mandatory control: ordinary Bayesian prior dependence must not be
presented as a distinctive neural defect. The deployment-prior oracle shows the
performance possible when the changed prior is known.

**Hidden intervention:** a nuisance direction selected on an independent discovery
split is removed without retraining. On untouched test pairs, it must reduce
nuisance response, preserve signal response and approach an analytically constrained
linear predictor's error. Random hidden directions, a nuisance-projected estimator
and a network trained with independent nuisances test whether the internal
intervention adds anything useful. Failure stops promotion of the correction.

**Identifiability:** target and nuisance signatures range from orthogonal to exactly
collinear. In the collinear case, spectral evidence cannot identify both amplitudes;
the prior can still produce a finite posterior. The unregularized estimator must
refuse this case rather than manufacture an estimate. These deliberately mixed
directions are statistical stress tests, not validated contamination physics.

All three comparisons are defined in the [protocol](docs/causal-reliability-protocol.md).
Approximate neural intervals use validation residuals, never test residuals. The
hidden intervention is coordinate-dependent and does not identify a unique circuit
or guarantee concept erasure. No novelty claim is made for neural retrieval,
sensitivity analysis or calibration; the protocol links the relevant prior work.

## Analytic controls

```python
from tierra_plus.diagnostics import gaussian_posterior, spectral_design

design = spectral_design(
    template=[0.020, 0.021, 0.019],
    sigma=[0.0001, 0.0001, 0.0001],
    overlap=0.0,
)
mean, covariance = gaussian_posterior(design, observations=[0.2, -0.3, 0.1], rho=0.9)
```

`observations` are centered, noise-whitened values. The prior has unit marginal
variances and correlation `rho`; the conditional observation-noise covariance is the identity.
`nuisance_projected_gls` provides the identifiable, prior-free reference, and
`constrained_linear_estimator` provides the best linear readout at a declared
nuisance-response cap. That frontier is not a universal bound on nonlinear models.

## Supporting spectral inference

The existing components remain available: validated `Spectrum` objects, bounded
uniform-prior `SpectralLikelihood`, seeded `NestedSampler` with weighted posterior
samples, top-hat tabulated-spectrum binning, supplied-intensity stellar mixing,
and public-asset provenance. The Gaussian causal oracle is separate from the
uniform-prior nested-sampling bridge.

```bash
uv run --extra inference python examples/synthetic_retrieval.py \
  --seed 42 --output /tmp/tierra-demo.json
```

The [synthetic retrieval tutorial](docs/tutorials/index.md) checks inference against
an analytic linear solution. Its [20-run repeated-noise check](examples/results/README.md)
is a software check with fixed truth, not prior-predictive calibration.

The [published WASP-39b PRISM check](examples/results/wasp39-prism/README.md)
uses the original CC-BY-4.0 observations and authors' fixed PICASO model. On 127
retained bins, the template plus a depth offset gives χ² = 247.891 for 126 degrees
of freedom; the flat null gives 1751.400. Both leave excess residual scatter under
the assumed independent errors. The template was selected using these data, so
this is a conditional, in-sample check, not a fresh retrieval or molecular detection.

The `tierra` and `petitRADTRANS` adapters remain deferred interface contracts and
fail explicitly. `tierra-plus retrieve ...` does not run the causal experiment.
NEO characterization is unimplemented and outside the active research direction.
Promotion to physical retrieval claims has explicit [research gates](BACKLOG.md).

## Verify and read

```bash
uv run ruff check .
uv run black --check .
uv run mypy src
uv run pytest -q
LC_ALL=C LANG=C uv run sphinx-build -M html docs docs/_build -W
uv run python scripts/bootstrap_public_data.py --verify-only
```

See the [documentation](docs/index.md), [theory and references](docs/theory/index.md),
[API](docs/api/index.md), and deferred [backend integration contract](docs/backend-integration.md).
For scientific outputs, retain input and source hashes, configuration, model weights,
split identities, raw predictions and failed checks. A passing software test does
not establish reliability on real observations.

Public-data downloads become visible at their destination only after checksum
verification; failed transfers are discarded and existing files are never replaced.
Use the locked development environment for patched tooling dependencies. Tagged
releases build without publishing credentials, then pass the built distributions
to a separate publishing job.

Code is Apache-2.0 licensed; see [LICENSE](LICENSE). Published spectral assets
retain their separate attribution and CC-BY-4.0 terms, including the
[prepared physical grids](examples/data/physical-grid/README.md).
