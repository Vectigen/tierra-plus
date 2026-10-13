# Statistical assumptions and references

The implemented likelihood is a normalized, diagonal Gaussian:

```{math}
\log p(y\mid\theta)=-\frac12\sum_i\left[
\frac{(y_i-m_i(\theta))^2}{\sigma_i^2}+\log(2\pi\sigma_i^2)\right].
```

`SpectralLikelihood` maps a unit cube to independent bounded uniform priors.
`NestedSampler` uses dynesty's weighted samples: posterior summaries must use
`result.weights`. Evidence uncertainty and effective sample size accompany the
result; an iteration cap does not certify convergence. Read sampler warnings.
See the [dynesty result and weight documentation](https://dynesty.readthedocs.io/en/stable/quickstart.html).

Fixed candidate-spectrum weights are proportional to likelihood times a normalized
model prior. These point hypotheses do not integrate over atmosphere parameters.
The legacy `log_evidences` field in that utility contains log joint weights,
not evidence for a fitted retrieval model.

For a quiet transit chord, the contamination operator multiplies planet-only depth by

```{math}
\epsilon_\lambda = \frac{I_{\mathrm{phot},\lambda}}
{(1-f_s-f_f)I_{\mathrm{phot},\lambda}+f_s I_{\mathrm{spot},\lambda}
+f_f I_{\mathrm{fac},\lambda}}.
```

All three intensity spectra must share the planet's wavelength sampling and units.
This excludes occulted active regions, limb dependence, and time-variable maps.
[Rackham, Apai & Giampapa (2018)](https://arxiv.org/abs/1711.05691) motivates this
transit-light-source model. [STCTM (Piaulet-Ghorayeb, 2025)](https://arxiv.org/abs/2508.19297)
is a fuller stellar-contamination retrieval framework; Tierra+ currently provides
only the mixing operator, not STCTM's atmosphere grids or stellar retrieval.

The synthetic example is an implementation check against a linear-Gaussian solution.
It is not simulation-based calibration. [Talts et al. (2018)](https://arxiv.org/abs/1804.06788)
describe prior-predictive simulation and rank diagnostics needed to assess Bayesian
computation over repeated datasets. No SBC coverage claim is made here.
