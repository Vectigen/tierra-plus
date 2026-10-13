# A reproducible synthetic retrieval

From a source checkout:

```bash
uv sync --dev --extra inference
uv run python examples/synthetic_retrieval.py --seed 42 --output /tmp/tierra-demo.json
```

The model is a baseline transit depth plus a Gaussian-shaped basis function centered
at 1.4 microns. Both coefficients have bounded uniform priors. Forty synthetic bins
have known independent Gaussian uncertainty. The wavelength is illustrative; the
feature is not a species identification or opacity table.

The output records the generated spectrum, injected parameters, priors, sampler
configuration and package versions, plus a canonical input hash. Compare
`posterior_mean` with `analytic_mean`, and inspect `mean_error_in_analytic_sigma`.
The analytic comparison assumes the posterior is far from prior boundaries.
`posterior_mean_chi_square` is a residual diagnostic, not a detection significance.

`posterior.samples` are weighted nested-sampling points. Use `posterior.weights`
for means, variances and posterior prediction. The output also records log evidence,
evidence error and effective sample size. Replaying the same software environment
and seed reproduces the input and sampler sequence; cross-version bitwise identity
is not promised.

The tests additionally compare a one-dimensional Gaussian posterior and known
evidence with a real dynesty run. No real-observation or atmospheric-backend
replication is included yet.

The source checkout also includes `examples/coverage_check.py` and complete repeated-noise
records under `examples/results/`. In 20 runs, nominal 90% intervals covered the
fixed baseline depth in 18 runs and feature amplitude in 16. The recorded Wilson
intervals are wide; the second parameter under-covers in this small experiment.
This is fixed-truth software validation, not prior-predictive SBC.
