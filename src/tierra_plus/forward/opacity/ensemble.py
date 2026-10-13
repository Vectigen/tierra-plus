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
    residual = np.asarray(observed.flux) - np.asarray(candidate.flux)
    sigma = np.asarray(observed.uncertainty)
    return float(-0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))


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
    prior_total = sum(prior_weights.values())
    log_evidences = {
        name: gaussian_log_likelihood(case.observed, candidate)
        + log(prior_weights[name] / prior_total)
        for name, candidate in case.candidates.items()
    }
    evidence_values = np.asarray(tuple(log_evidences.values()))
    max_evidence = float(np.max(evidence_values))
    normalization = float(max_evidence + np.log(np.sum(np.exp(evidence_values - max_evidence))))
    weights = {
        name: float(np.exp(log_evidence - normalization))
        for name, log_evidence in log_evidences.items()
    }
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
