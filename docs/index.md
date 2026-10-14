# Tierra+ documentation

Tierra+ studies the causal reliability of spectral inverse models: when a predictor
becomes confidently wrong after a nuisance or physical-model change, and which
corrections survive explicit controls and held-out physical states.

Start with the [frozen experiment protocol](causal-reliability-protocol.md). The first
experiment generates synthetic observations and labels on a public spectral grid,
using a known target shape and explicit nuisance geometry. Exact Bayesian controls
separate ordinary prior dependence from approximation error; deliberately collinear
signatures expose cases where the data cannot identify the target. The experiment
does not establish a molecular detection or a failure of a physical retrieval.

The [physical-grid protocol](physical-grid-protocol.md) separately trains a
metallicity estimator on published PICASO atmospheres. It compares nuisance
augmentation with matched ridge and exact finite-grid references, then challenges
all methods with photochemical spectra. Its bounded augmentation benefit does not
survive that change in physics; discrete-label uncertainty cannot be interpreted
as continuous atmospheric precision.

```{toctree}
:maxdepth: 2
:caption: Contents

causal-reliability-protocol
physical-grid-protocol
theory/index
tutorials/index
api/index
backend-integration
```

From a source checkout:

```bash
uv sync --dev
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --with h5py==3.16.0 python -m examples.causal_reliability \
  --config examples/configs/causal-reliability-v1.json \
  --output /tmp/tierra-causal
```

The existing [retrieval tutorial](tutorials/index.md) and [statistical theory](theory/index.md)
document the supporting spectral likelihood, weighted nested-sampling output,
stellar intensity mixing and provenance tools. Install the `inference` extra for
that sampling example. Atmospheric backend contracts are deferred; NEO inference
is unimplemented and outside the active direction.
