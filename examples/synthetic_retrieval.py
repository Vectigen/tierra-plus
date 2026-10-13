"""Seeded two-parameter spectral retrieval with an analytic regression check.

Run: uv run --extra inference python examples/synthetic_retrieval.py --output /tmp/tierra-demo.json
The Gaussian feature is an illustrative basis, not a radiative-transfer calculation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tierra_plus.inference import NestedSampler, NestedSamplerConfig
from tierra_plus.inference.spectral import SpectralLikelihood, UniformPrior
from tierra_plus.models import Spectrum
from tierra_plus.provenance import content_hash


def run(seed: int = 42) -> dict[str, object]:
    """Recover a linear template and record inputs, weighted posterior, and checks."""
    wavelength = np.linspace(1.0, 2.0, 40)
    feature = np.exp(-0.5 * ((wavelength - 1.4) / 0.08) ** 2)
    design = np.column_stack([np.ones(40), feature])
    truth = np.array([0.01, 0.001])
    sigma = 0.0001
    observed = Spectrum(
        tuple(wavelength),
        tuple(design @ truth + np.random.default_rng(seed).normal(0, sigma, 40)),
        (sigma,) * 40,
    )
    priors = (
        UniformPrior("baseline_depth", 0.008, 0.012),
        UniformPrior("feature_amplitude", -0.001, 0.003),
    )
    bridge = SpectralLikelihood(
        observed, lambda theta: Spectrum(tuple(wavelength), tuple(design @ theta)), priors
    )
    config = NestedSamplerConfig(nlive=150, dlogz=0.05, seed=seed)
    result = NestedSampler(config=config, model_bridge=bridge).run()
    samples, weights = np.asarray(result.samples), np.asarray(result.weights)
    mean = np.sum(samples * weights[:, None], axis=0)
    covariance = np.linalg.inv(design.T @ design / sigma**2)
    analytic_mean = covariance @ design.T @ np.asarray(observed.flux) / sigma**2
    predictive = design @ mean
    inputs = {
        "observed": observed.to_dict(),
        "priors": [asdict(p) for p in priors],
        "sampler": asdict(config),
        "truth": truth.tolist(),
        "model": "linear_gaussian_feature_v1",
    }
    return {
        "kind": "synthetic_software_validation",
        "inputs": inputs,
        "input_sha256": content_hash(inputs),
        "versions": {name: version(name) for name in ("tierra-plus", "numpy", "dynesty")},
        "posterior": asdict(result),
        "posterior_mean": mean.tolist(),
        "analytic_mean": analytic_mean.tolist(),
        "mean_error_in_analytic_sigma": (
            (mean - analytic_mean) / np.sqrt(np.diag(covariance))
        ).tolist(),
        "posterior_mean_chi_square": float(
            np.sum(((np.asarray(observed.flux) - predictive) / sigma) ** 2)
        ),
        "limitations": (
            "Independent Gaussian noise; synthetic linear feature; "
            "no atmospheric or NEO validation; weighted points are not iid posterior samples."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args.seed), indent=2, allow_nan=False) + "\n")
    print(args.output)
