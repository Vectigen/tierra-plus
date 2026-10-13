# Tierra+

**Reproducible inference components for spectral retrieval research.**

Tierra+ connects validated spectra and explicit priors to a seeded nested sampler,
with weighted posterior output, evidence estimates, provenance hashes, and a simple
stellar-contamination operator. It includes a runnable synthetic retrieval and
analytic checks. Atmospheric radiative-transfer adapters and NEO inference remain
unfinished; this is research software, not a validated mission pipeline.

## Reproduce the working example

```bash
git clone https://github.com/Vectigen/tierra-plus.git
cd tierra-plus
uv sync --dev --extra inference
uv run pytest -q
uv run python examples/synthetic_retrieval.py --seed 42 --output /tmp/tierra-demo.json
```

The example generates a noisy linear spectral feature, fits its depth and amplitude,
and compares the weighted posterior mean with analytic weighted least squares.
Its JSON output contains inputs, priors, seed, package versions, input hash,
weighted samples, evidence uncertainty, effective sample size and residual checks.
It uses no API, external data or paid compute. The spectral feature is a mathematical
test model, **not a molecular opacity or radiative-transfer calculation**.

## What works

- `Spectrum` validates finite flux, positive uncertainties and an increasing wavelength grid.
- `SpectralLikelihood` combines a callable spectrum model with explicit bounded uniform priors.
- `NestedSampler` runs optional `dynesty`, preserving normalized posterior weights,
  evidence uncertainty and reproducible RNG state. Its points are not equal-weight posterior draws.
- `contaminate_transit` applies supplied spot/facula intensity spectra to a quiet-chord transit model.
- Public-asset manifests verify SHA-256 hashes and constrain asset paths to the data root.
- Fixed-spectrum hypothesis weighting, run logs and replay-manifest primitives are available.

## Python API

```python
from tierra_plus.models import Spectrum
from tierra_plus.inference import NestedSampler, NestedSamplerConfig
from tierra_plus.inference.spectral import SpectralLikelihood, UniformPrior

observed = Spectrum([1.0, 1.1], [0.0100, 0.0102], [0.0001, 0.0001])
bridge = SpectralLikelihood(
    observed,
    model=lambda theta: Spectrum([1.0, 1.1], [theta[0], theta[0]]),
    priors=[UniformPrior("transit_depth", 0.005, 0.015)],
)
result = NestedSampler(
    config=NestedSamplerConfig(seed=42), model_bridge=bridge,
).run()
```

Use a scientifically justified forward model and likelihood for real spectra.
Independent Gaussian errors and independent uniform priors are explicit assumptions
of this bridge. Correlated noise, instrument response and atmospheric systematics
are not handled implicitly.

## Scientific scope

| Component | Status |
| --- | --- |
| Seeded nested sampling | Executable; real-dynesty Gaussian posterior/evidence test |
| Spectral retrieval example | Synthetic linear feature with analytic comparison |
| Stellar contamination | Algebraic intensity mixing; requires user-supplied stellar spectra |
| Opacity selection | Fixed candidate spectra; no parameter-marginalized opacity comparison |
| `tierra`, `petitRADTRANS` adapters | Interface contracts only; fail explicitly until integrated |
| NEO characterization | Not implemented |
| JWST replication, SBC, detection claims | Not established by this repository |

`uv run tierra-plus retrieve ...` exercises the unresolved backend interface and
returns a configuration error; use the Python API/example for the implemented lane.
The `inference` extra is required for actual sampling. Missing dependencies do not
silently trigger surrogate physics.

## Verify and read

```bash
uv run ruff check .
uv run black --check .
uv run mypy src
uv run pytest -q
LC_ALL=C LANG=C uv run sphinx-build -M html docs docs/_build -W
uv run python scripts/bootstrap_public_data.py --verify-only
```

See [theory and references](docs/theory/index.md), [tutorial](docs/tutorials/index.md),
[API](docs/api/index.md), and [backend integration contract](docs/backend-integration.md).
For scientific outputs, record input data hashes, priors, backend/opacity versions,
sampler settings and calibration diagnostics. A passing software example alone
does not establish calibration on real observations.

Apache-2.0 licensed. See [LICENSE](LICENSE).
