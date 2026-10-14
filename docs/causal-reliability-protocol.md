# Causal reliability of spectral inverse models

**Question: does an inverse model use spectral evidence, or a nuisance correlation
that stops holding at deployment?**

This is Tierra's focused research direction. The target is a failure we can cause,
measure and distinguish from ordinary Bayesian prior dependence. The first experiment
is a controlled model organism, anchored to public spectral assets. It does not test
an existing atmospheric neural retrieval or establish a molecular detection.

## Frozen first experiment

The machine-readable specification is
[`causal-reliability-v1.json`](../examples/configs/causal-reliability-v1.json).
It is committed before running the main experiment. These thresholds are project
acceptance criteria, not astrophysical standards. Changes after examining results
require a new protocol and separately identified experiment.

Use the existing hashed WASP-39 PRISM wavelength grid, uncertainties and original
authors' PICASO template, retaining the 127 bins above the preselected 2.3-micrometre
cut. In noise-whitened coordinates, remove the template's component parallel to a
constant depth offset and normalize it to a unit vector `s`. Let `u` be the unit
constant-depth direction in those coordinates. Define

```text
target a, independent latent z, measurement noise epsilon ~ standard Normal
b = rho*a + sqrt(1-rho^2)*z
c = overlap*s + sqrt(1-overlap^2)*u
x = 2*a*s + 6*b*c + epsilon
synthetic_depth = fixed_reference_depth + published_sigma*x
```

All generated observations and labels are synthetic. `a` is a standardized template
amplitude, not an abundance. Overlap zero describes an additive depth offset after
the declared centering. Overlaps 0.95 and 1 deliberately mix that offset with the
target shape; they are controlled identifiability challenges, not validated instrument
or stellar-contamination models. At overlap 1, spectral evidence cannot separately
identify both amplitudes without a prior. A method must expose this limitation.

The data split sizes are 8192 training, 1024 validation, 1024 discovery and 4096 test
objects. Training seeds are 11, 23, 42, 71 and 101. Split generators are independently
derived from the recorded data seed and run identity. Test distributions have
correlations +0.9, 0 and -0.9; they reuse the same target, independent latent and
noise within a run, making distribution comparisons paired. No test outputs select
models, intervention directions, thresholds or hyperparameters.

## Hypotheses and decisions

**H1: good in-distribution prediction can conceal nuisance dependence that becomes
unreliable after a prior change.** At overlap zero, require the learned model's IID
RMSE to be at most 1.25 times the matched-prior Bayes RMSE. A qualifying failure has
reversed-prior RMSE at least 1.5 times IID RMSE and a Wilson upper confidence limit
below 0.85 for its nominal 90% interval coverage. Require this in at least four of
five training runs. A failed training-quality gate makes the learned-model diagnosis
inconclusive; poor fitting is not evidence of an interesting learned mechanism.

Prior dependence is not automatically a neural defect. Compare each network with
the **exact correlated-training-prior Bayes estimator**, including its paired
nuisance response, and with the **exact deployment-prior Bayes estimator**. Report
the RMS discrepancy between the network's intervention effect and the training
prior oracle's effect. A discrepancy at most 0.1 target standard deviations is
evidence that the oracle explains the effect to that tolerance, not evidence of a
distinctive neural failure. This is an operational tolerance, not a statistical
equivalence proof. Ordinary random-split prediction accuracy cannot settle this.

**H2: removing a discovered hidden nuisance direction is a useful correction.** Fit
one direction from paired activation changes on the independent discovery split.
The direction is the zero-intercept least-squares slope of the single tanh layer's
paired activation differences against the nuisance change, normalized in Euclidean
coordinates. Center on the pooled discovery activation mean. Remove its centered
projection without retraining. This coordinate-dependent operation is not a
guaranteed concept-erasure procedure. On untouched test pairs, require
at least an 80% reduction in RMS nuisance response, IID RMSE within 10% of the
optimal constrained **linear** predictor at a nuisance-response cap of 20% of the
training-prior Bayes response, and retention of at least 80% of the unmodified
model's signed signal response.
Require all three conditions in at least four of five overlap-zero runs. Also
compare against ten fixed random hidden directions, a nuisance-projected estimator
and a network trained with independent nuisances. If a simple baseline matches or
improves the error/sensitivity tradeoff, do not claim a special benefit from the
hidden intervention. A failed H2 is a stop condition for promoting this correction. The comparison is
against a linear reference, not a proven bound for nonlinear predictors under a
finite-endpoint intervention. The unmodified nuisance RMS and positive signed
signal response must both exceed 1e-6; otherwise relative gates are inconclusive.

An initial design review rejected a <=10% loss relative to the original IID model:
the correlated prior gives useful information, so removing its influence entails an
error tradeoff even for an optimal linear estimator. The constrained comparator
makes that tradeoff explicit before evaluating the hidden intervention.

The internal intervention tests sufficiency for changing predictions on the stated
pairs; it does not establish a unique circuit, semantic feature, or necessary causal
mechanism. Even successful nuisance erasure can destroy genuine information in the
high-overlap cases. Those cases must remain in the report.

## Baselines and measurements

- Exact Gaussian conditioning under the correlated training prior and each actual
  test prior. The latter is an oracle with knowledge unavailable to an ordinary
  deployed model; report it as such.
- Nuisance-projected generalized least squares, using the known design matrix. It
  is unbiased and invariant when identifiable, but has high variance near degeneracy
  and must refuse the rank-deficient case.
- The analytically optimal linear predictor subject to an absolute nuisance-response
  cap; report its expected MSE and empirical test error. Its known-design/prior access
  is privileged. It does not establish a global bound for arbitrary nonlinear models.
- Ridge regression, with a fixed penalty of 1, trained on the same examples.
- A one-hidden-layer tanh network, 32 units, 800 Adam steps, batch size 256 and
  learning rate 0.003; a same-budget network trained with independent nuisances.
- The frozen network after discovered-direction removal and ten seeded random
  equal-rank removals. Keep all random-control outcomes, not only the most flattering.

Neural and ridge intervals use a single residual standard deviation estimated on
their IID validation data, separately for each predictor after any direction is
frozen. No uncertainty scale is fitted on test residuals. They are approximate
Gaussian predictive intervals,
not learned full posteriors; calibration cannot be assumed. Report RMSE, signed bias,
Gaussian NLL, 90% interval width, coverage and its Wilson interval separately. Exact
Bayes intervals use the analytic posterior covariance. GLS uses its known noise
variance. Do not pool dependent trials across training seeds as independent evidence.

Discovery pairs set `b` to -0.5 and +0.5. Primary held-out nuisance pairs set `b` to -1 and +1 with the same `a` and noise.
A secondary held-out nuisance magnitude uses -2 and +2; keep it separate from the
primary gate. The signal intervention sets `a` to -1 and +1 with the same independently drawn `b`
and noise. Record signed response per unit intervention and RMS response. Removing
the entire target component is an additional diagnostic, not an estimate of a real
molecule's removal. It is a shift in the input-generating mechanism, so uncertainty
from the original likelihood is not automatically valid there.

```text
verified public spectral assets
        |
        v
declared signal/nuisance geometry --> analytic Bayes + GLS controls
        |
independent train / validation / discovery / test RNG streams
        |          |             |               |
   fit weights   residual SD   choose direction   paired interventions
        +----------+-------------+---------------+
                                                  |
                                       frozen metrics and decisions
                                                  |
                                  raw arrays, weights, hashes, figures
```

Tests cover Gaussian conditioning, paired-data invariants, strict split isolation,
rank-deficient refusal, model gradients, deterministic fitting, hidden interventions,
metric accounting and artifact replay. Invalid or nonfinite inputs fail explicitly.
No remote provider, GPU or paid service is required.

## Why this direction

[Nanda et al.'s pragmatic interpretability approach](https://www.neelnanda.io/vision)
emphasizes objective proxy tasks, intervention predictions and simple baselines.
Here the proxy is the error/sensitivity tradeoff under a known spectral intervention.
[His patching discussion](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
also motivates keeping claims conditional on the chosen intervention distribution.

Spectral-network sensitivity analysis already exists:
[Yip et al.](https://arxiv.org/abs/2011.11284). Atmospheric neural posterior estimation
with calibration diagnostics also exists:
[Vasist et al.](https://arxiv.org/abs/2301.06575). This project does not claim novelty
for saliency maps, neural retrieval, or calibration. Its immediate contribution is
an executable falsification protocol separating prior shift, approximation error,
and identifiability, with cheap controls that can disqualify an elaborate explanation.

## Scope and next promotion gate

Reuse the existing spectral validation, binning and public-asset provenance. Keep
the existing nested-sampling APIs as support tools; their uniform-prior bridge is
not silently repurposed as a correlated Gaussian posterior. A tiny separate analytic
module supplies that oracle.

Full radiative transfer, fresh molecule-specific opacities, NEOs, large neural
posterior models and claims about actual atmospheric retrieval failures are outside
this first experiment. They are not required to falsify the stated claims. Promotion
beyond a model organism requires repeating the intervention audit on a separately
trained physical simulation grid with defensible nuisance transformations. If the
simple baselines suffice here, improve that next empirical test instead of adding
more interpretability machinery.
