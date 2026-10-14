# API reference

- `tierra_plus.diagnostics.spectral_design(template, sigma, overlap, ...)`:
  noise-whitened synthetic target/nuisance geometry with explicit shape overlap.
- `tierra_plus.diagnostics.gaussian_posterior(design, observations, rho=...)`:
  exact Gaussian conditioning with a correlated unit-variance prior.
- `tierra_plus.diagnostics.nuisance_projected_gls(design, observations)`:
  prior-free target estimate and variance; refuses non-identifiable designs.
- `tierra_plus.diagnostics.constrained_linear_estimator(design, rho, max_nuisance_response)`:
  optimal linear readout and expected MSE under a nuisance-response cap.
- `tierra_plus.diagnostics.network.fit(x, y, seed=..., ...)`:
  seeded NumPy tanh regressor with training-only normalization, explicit hidden
  activations, paired direction discovery and reloadable numerical state.
- `tierra_plus.models.Spectrum(wavelength_microns, flux, uncertainty=None)`:
  immutable spectrum values with finite-array and grid validation.
- `tierra_plus.inference.spectral.UniformPrior(name, lower, upper)`:
  finite ordered bounds for one parameter.
- `tierra_plus.inference.spectral.SpectralLikelihood(observed, model, priors)`:
  prior transform and normalized diagonal Gaussian likelihood for a callable model.
- `tierra_plus.inference.NestedSampler(config=..., model_bridge=...)`:
  optional dynesty sampler; `run()` returns weighted samples, log likelihoods,
  log evidence/error and metadata. A missing backend raises an explicit error.
- `tierra_plus.contamination.contaminate_transit(planet, photosphere, spots, faculae,
  spot_fraction=0, facula_fraction=0)`: quiet-chord stellar intensity mixing.
- `tierra_plus.forward.opacity.ensemble.posterior_weights_for_case(case, priors=None)`:
  normalized weights for fixed spectra, not fitted-model marginal likelihoods.
  Extreme positive uncertainties use log-space normalization; an all-zero
  representable likelihood fails explicitly instead of selecting a NaN winner.
- `tierra_plus.provenance`: content hashing, append-only run-log and replay-manifest helpers.
  Structured hashes require string mapping keys and finite JSON numbers. File-byte
  SHA256 and structured-payload digests retain separate identity namespaces.

See the source docstrings for parameter definitions, the
[causal protocol](../causal-reliability-protocol.md) for experimental assumptions,
and the [tutorial](../tutorials/index.md) for the supporting nested-sampling run.
