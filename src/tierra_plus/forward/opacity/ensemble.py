"""Bayesian model averaging utilities for opacity ensembles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import log
from pathlib import Path

import numpy as np

from tierra_plus.forward.opacity import get_opacity_model, verify_opacity_asset
from tierra_plus.models import Spectrum


@dataclass(frozen=True)
class BenchmarkCase:
    """Synthetic or public benchmark case used to compare opacity models."""

    case_id: str
    description: str
    observed: Spectrum
    candidates: Mapping[str, Spectrum]
    expected_best_model: str


@dataclass(frozen=True)
class EnsembleWeightResult:
    """Posterior weights over candidate opacity models."""

    weights: Mapping[str, float]
    log_evidences: Mapping[str, float]
    best_model: str


def gaussian_log_likelihood(observed: Spectrum, candidate: Spectrum) -> float:
    """Compute a diagonal-Gaussian log likelihood for one candidate spectrum.

    Parameters
    ----------
    observed
        Observed spectrum with uncertainties.
    candidate
        Candidate model prediction on the same wavelength grid.
    """

    if observed.wavelength_microns != candidate.wavelength_microns:
        raise ValueError("Observed and candidate spectra must share the same wavelength grid.")
    if observed.uncertainty is None:
        raise ValueError("Observed spectrum must include uncertainties for likelihood evaluation.")
    sigma = np.asarray(observed.uncertainty)
    # Squaring sigma can underflow to zero for valid positive uncertainties.
    # A representably infinite squared residual means zero likelihood, not NaN.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = np.asarray(observed.flux) - np.asarray(candidate.flux)
        standardized = residual / sigma
        # Opposite, very large finite fluxes can overflow their difference even
        # when division by a large uncertainty would give a finite residual.
        overflowed = np.isinf(residual)
        standardized[overflowed] = (
            np.asarray(observed.flux)[overflowed] / sigma[overflowed]
            - np.asarray(candidate.flux)[overflowed] / sigma[overflowed]
        )
        value = float(-0.5 * np.sum(standardized**2 + np.log(2.0 * np.pi) + 2.0 * np.log(sigma)))
    if np.isnan(value) or value == np.inf:
        raise ValueError("Spectrum scales do not give a valid Gaussian log likelihood")
    return value


def posterior_weights_for_case(
    case: BenchmarkCase,
    priors: Mapping[str, float] | None = None,
) -> EnsembleWeightResult:
    """Weight fixed candidate spectra using Gaussian likelihoods and model priors.

    A fixed spectrum is a point hypothesis: this does not marginalize atmospheric
    parameters and must not be interpreted as evidence for a fitted model.
    """

    prior_weights = priors if priors is not None else {name: 1.0 for name in case.candidates}
    if not case.candidates or set(prior_weights) != set(case.candidates):
        raise ValueError("Priors must match a nonempty set of candidates")
    if any(not np.isfinite(v) or v <= 0 for v in prior_weights.values()):
        raise ValueError("Model prior weights must be finite and strictly positive")
    log_priors = {name: log(weight) for name, weight in prior_weights.items()}
    prior_max = max(log_priors.values())
    log_prior_total = prior_max + log(sum(np.exp(v - prior_max) for v in log_priors.values()))
    log_likelihoods = {
        name: gaussian_log_likelihood(case.observed, candidate)
        for name, candidate in case.candidates.items()
    }
    log_evidences = {
        name: likelihood + log_priors[name] - log_prior_total
        for name, likelihood in log_likelihoods.items()
    }
    evidence_values = np.asarray(tuple(log_evidences.values()))
    if np.any(np.isnan(evidence_values) | (evidence_values == np.inf)) or not np.any(
        np.isfinite(evidence_values)
    ):
        raise ValueError("At least one candidate must have finite log evidence")
    # Center likelihoods before adding priors: at very large negative scales,
    # adding a small log prior first would round away real prior differences.
    likelihood_values = np.asarray(tuple(log_likelihoods.values()))
    with np.errstate(over="ignore"):
        log_scores = likelihood_values - np.max(likelihood_values)
    log_scores += np.asarray(tuple(log_priors[name] - prior_max for name in log_likelihoods))
    unnormalized = np.exp(log_scores - np.max(log_scores))
    normalized = unnormalized / unnormalized.sum()
    weights = {name: float(weight) for name, weight in zip(log_evidences, normalized, strict=True)}
    best_model = max(weights, key=lambda name: weights[name])
    return EnsembleWeightResult(weights=weights, log_evidences=log_evidences, best_model=best_model)


def load_benchmark_case(identifier: str, data_root: Path | None = None) -> BenchmarkCase:
    """Load a benchmark case from the tracked public data manifest."""

    asset = get_opacity_model(identifier, data_root=data_root)
    path = verify_opacity_asset(asset, data_root=data_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = Spectrum.from_dict(payload["observed"])
    candidates = {
        name: Spectrum(
            wavelength_microns=list(observed.wavelength_microns),
            flux=list(flux_values),
            uncertainty=list(observed.uncertainty) if observed.uncertainty is not None else None,
        )
        for name, flux_values in payload["candidates"].items()
    }
    return BenchmarkCase(
        case_id=str(payload["case_id"]),
        description=str(payload["description"]),
        observed=observed,
        candidates=candidates,
        expected_best_model=str(payload["expected_best_model"]),
    )
