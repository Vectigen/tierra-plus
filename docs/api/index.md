# API reference

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
- `tierra_plus.provenance`: content hashing, append-only run-log and replay-manifest helpers.

See the source docstrings for parameter definitions and the tutorial for a complete run.
