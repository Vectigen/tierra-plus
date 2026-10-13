# Tierra+ documentation

Tierra+ provides spectral likelihoods, seeded nested sampling, weighted posterior
output, stellar intensity mixing, and provenance utilities for retrieval research.
The synthetic example has analytic validation; atmospheric backend adapters and
NEO inference are not implemented.

```{toctree}
:maxdepth: 2
:caption: Contents

backend-integration
theory/index
tutorials/index
api/index
```

Install with `uv sync --dev --extra inference`, run `uv run pytest -q`, and follow
the tutorial for an executable local retrieval. Read the theory page for assumptions
and the distinction between software checks and scientific validation.
