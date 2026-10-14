# Tierra+ documentation

Tierra+ studies the causal reliability of spectral inverse models: when a predictor
uses a training correlation that becomes misleading after deployment, and whether
a targeted hidden intervention improves that error/sensitivity tradeoff.

Start with the [frozen experiment protocol](causal-reliability-protocol.md). The first
experiment generates synthetic observations and labels on a public spectral grid,
using a known target shape and explicit nuisance geometry. Exact Bayesian controls
separate ordinary prior dependence from approximation error; deliberately collinear
signatures expose cases where the data cannot identify the target. The experiment
does not establish a molecular detection or a failure of a physical retrieval.

```{toctree}
:maxdepth: 2
:caption: Contents

causal-reliability-protocol
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
