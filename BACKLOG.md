# Research promotion gates

The active question is [causal reliability of spectral inverse models](docs/causal-reliability-protocol.md).
Its first experiment is a controlled synthetic problem with a public spectral shape.
Passing software tests or reproducing that experiment does not clear any of the
physical-validation gates below. A failed hidden-direction correction remains a
negative result; it does not justify adding more interpretability machinery until
the simple controls are understood.

## 1. Move from controlled geometry to a physical simulation grid

Use a licensed, versioned atmosphere grid with hashed assets and independently
varying target and nuisance parameters. Declare the physical intervention, bin
response and observation-noise assumptions before fitting. Hold out physical
parameter combinations, not only random noise realizations of the same spectra.

Repeat the paired audit with matched-prior and deployment-prior references where
tractable, an independent-nuisance training control, and explicit ambiguous cases.
Promotion requires evidence that the intervention preserves the intended target
while changing a defensible nuisance. A synthetic template amplitude cannot be
renamed as abundance without this step.

## 2. Audit an independently trained retrieval

Freeze an external retrieval model, its training distribution and its preprocessing.
Specify a falsifiable prediction about nuisance response before inspecting held-out
interventions. Compare the intervention effect with ordinary prior dependence and
approximation error. Retain failures and test whether a simple estimator or changed
training distribution accounts for the effect before attributing it to a hidden
mechanism. Claim only what the independently evaluated model and data support.

## 3. Extend uncertainty beyond one validation residual scale

Add posterior or conditional uncertainty estimates only with explicit assumptions
and held-out calibration tests. Separate prior-predictive computational calibration
from robustness under a changed prior or observation model. For correlated noise,
validate the covariance and whitening against a known reference before interpreting
coverage; never repair coverage by fitting an uncertainty scale on test residuals.

## Deferred legacy scope

Full `tierra` and `petitRADTRANS` integration, molecule-specific opacity comparison,
joint stellar-atmosphere retrieval and NEO characterization are deferred. Their
interface contracts or placeholder modules are not implemented capabilities. They
should return to active scope only when needed for a specific promotion gate and
accompanied by independent validation.

The existing synthetic nested-sampling example and conditional WASP-39b check remain
supporting inference tests. They do not satisfy the promotion gates above.
