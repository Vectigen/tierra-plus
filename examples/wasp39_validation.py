"""Conditional fixed-template check on published WASP-39b JWST PRISM data.

Run with: uv run --extra inference --with h5py==3.16.0 --with matplotlib \
    python examples/wasp39_validation.py --output /tmp/tierra-wasp39

The PICASO template was fitted to these observations by the original authors.
This checks data handling and conditional inference, not independent atmospheric
retrieval, a molecular detection, or out-of-sample prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import chi2

from tierra_plus.forward.tabulated import bin_tabulated_spectrum
from tierra_plus.inference import NestedSampler, NestedSamplerConfig
from tierra_plus.inference.spectral import SpectralLikelihood, UniformPrior
from tierra_plus.models import Spectrum
from tierra_plus.provenance import content_hash

DATA = Path(__file__).parent / "data" / "wasp39-prism"
HASHES = {
    "FIREFLy_transit_spec.h5": "02fa509fc8cd81eaa8451a378a72cb3df7488cfcf5ea83c3b74c978f978d884c",
    "best_fit_model.h5": "108aa6aef6d3b21e23cd4f5b32f7150f420914b8267e03e5cb92d9c2c1d3b61c",
}


def load_inputs() -> tuple[Spectrum, np.ndarray, dict[str, object]]:
    """Verify original bytes, preserve depth units, and use all bins above 2.3 um."""
    for filename, expected in HASHES.items():
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published input checksum mismatch: {filename}")
    with h5py.File(DATA / "FIREFLy_transit_spec.h5") as handle:
        wave = handle["wavelength"][:]
        depth = handle["transit_depth"][:]
        uncertainty = handle["transit_depth_uncertainty"][:]
    with h5py.File(DATA / "best_fit_model.h5") as handle:
        model_wave, model_depth = handle["wavelength"][:], handle["transit_depth"][:]
    order = np.argsort(model_wave)
    template = Spectrum(tuple(model_wave[order]), tuple(model_depth[order]))
    # Published HDF5 contains centers, not edges. State this response approximation.
    edges = np.concatenate(
        (
            [wave[0] - (wave[1] - wave[0]) / 2],
            (wave[:-1] + wave[1:]) / 2,
            [wave[-1] + (wave[-1] - wave[-2]) / 2],
        )
    )
    binned = np.asarray(bin_tabulated_spectrum(template, edges).flux)
    # The publication warns about partial saturation at shorter wavelengths.
    # Fix this wavelength cut before fitting; do not select bins by residuals.
    keep = edges[:-1] >= 2.3
    observed = Spectrum(tuple(wave[keep]), tuple(depth[keep]), tuple(uncertainty[keep]))
    inputs = {
        "source_doi": "10.5281/zenodo.7388032",
        "publication_doi": "10.1038/s41586-022-05677-y",
        "license": "CC-BY-4.0",
        "files_sha256": HASHES,
        "wavelength_unit": "micrometre",
        "flux_quantity": "dimensionless transit depth (Rp/Rstar)^2",
        "uncertainty": "published 1-sigma depth errors, assumed independent",
        "selection": "all midpoint bins with lower edge >= 2.3 micrometre",
        "selected_original_indices": np.flatnonzero(keep).tolist(),
        "original_bin_count": len(wave),
        "selected_bin_count": int(keep.sum()),
        "bin_edges_um": np.column_stack([edges[:-1][keep], edges[1:][keep]]).tolist(),
        "bin_response": "uniform top-hat; edges inferred from adjacent centers",
        "observed": observed.to_dict(),
        "picaso_template_depth": binned[keep].tolist(),
    }
    return observed, binned[keep], inputs


def fit_offset(observed: Spectrum, template: np.ndarray, prior: UniformPrior, seed: int) -> dict:
    """Fit one additive depth parameter and compare to Gaussian quadrature."""
    y, sigma = np.asarray(observed.flux), np.asarray(observed.uncertainty)
    precision = 1 / sigma**2
    mean = float(np.sum((y - template) * precision) / precision.sum())
    sd = float(precision.sum() ** -0.5)
    bridge = SpectralLikelihood(
        observed,
        lambda theta: Spectrum(observed.wavelength_microns, tuple(template + theta[0])),
        [prior],
    )
    config = NestedSamplerConfig(nlive=200, dlogz=0.03, seed=seed)
    posterior = NestedSampler(config=config, model_bridge=bridge).run()
    samples, weights = np.asarray(posterior.samples)[:, 0], np.asarray(posterior.weights)
    sampled_mean = float(np.sum(samples * weights))
    mass = (
        math.erf((prior.upper - mean) / sd / math.sqrt(2))
        - math.erf((prior.lower - mean) / sd / math.sqrt(2))
    ) / 2
    if mass < 0.999999:
        raise ValueError("Analytic untruncated-moment comparison requires negligible truncation")
    analytic_logz = (
        bridge.log_likelihood([mean])
        + math.log(sd * math.sqrt(2 * math.pi))
        - math.log(prior.upper - prior.lower)
        + math.log(mass)
    )
    best_residual = (y - template - mean) / sigma
    statistic = float(np.sum(best_residual**2))
    # Conditional posterior predictive check, with parameter uncertainty retained.
    rng = np.random.default_rng(seed + 1)
    draws = rng.choice(samples, size=2048, p=weights)
    observed_discrepancy = np.sum(((y - template - draws[:, None]) / sigma) ** 2, axis=1)
    replicated_discrepancy = np.sum(rng.normal(size=(2048, len(y))) ** 2, axis=1)
    exceed = int(np.sum(replicated_discrepancy >= observed_discrepancy))
    return {
        "prior": asdict(prior),
        "sampler": asdict(config),
        "posterior": asdict(posterior),
        "posterior_mean": sampled_mean,
        "analytic_mean": mean,
        "analytic_sd": sd,
        "analytic_prior_mass": mass,
        "mean_error_in_analytic_sigma": (sampled_mean - mean) / sd,
        "analytic_log_evidence": analytic_logz,
        "sampled_minus_analytic_log_evidence": posterior.log_evidence - analytic_logz,
        "chi_square_at_analytic_mean": statistic,
        "degrees_of_freedom": len(y) - 1,
        "chi_square_survival_probability": float(chi2.sf(statistic, len(y) - 1)),
        "standardized_residuals": best_residual.tolist(),
        "posterior_predictive": {
            "seed": seed + 1,
            "draws": 2048,
            "exceedances": exceed,
            "tail_fraction": exceed / 2048,
        },
        "prediction_at_posterior_mean": (template + sampled_mean).tolist(),
    }


def run(seed: int = 42) -> dict:
    observed, template, inputs = load_inputs()
    # Same 0.01-wide nuisance-depth prior; both likelihood peaks lie far from bounds.
    flat = fit_offset(observed, np.zeros(len(template)), UniformPrior("depth", 0.015, 0.025), seed)
    picaso = fit_offset(observed, template, UniformPrior("depth_offset", -0.005, 0.005), seed)
    return {
        "kind": "real_data_conditional_fixed_template_check",
        "inputs": inputs,
        "input_sha256": content_hash(inputs),
        "versions": {p: version(p) for p in ("tierra-plus", "numpy", "dynesty", "scipy", "h5py")},
        "fits": {"flat_depth": flat, "published_picaso_plus_offset": picaso},
        "conditional_delta_log_evidence": (
            picaso["posterior"]["log_evidence"] - flat["posterior"]["log_evidence"]
        ),
        "limitations": [
            "The fixed PICASO template was selected using these data: this is an in-sample check.",
            "Conditional evidence does not marginalize atmospheric parameters or model selection.",
            "Midpoint top-hat bins approximate the bin response, without throughput weighting.",
            "Independent published errors omit covariance and pipeline/model systematics.",
            "No fresh radiative-transfer calculation, chemical retrieval, or molecular detection.",
        ],
    }


def plot_result(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = result["inputs"]["observed"]
    wave = observed["wavelength_microns"]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), constrained_layout=True)
    axes[0].errorbar(
        wave,
        np.asarray(observed["flux"]) * 1e6,
        yerr=np.asarray(observed["uncertainty"]) * 1e6,
        fmt=".",
        color="black",
        alpha=0.7,
        label="Published FIREFLy depths",
    )
    for name, fit in result["fits"].items():
        axes[0].plot(
            wave,
            np.asarray(fit["prediction_at_posterior_mean"]) * 1e6,
            label=name.replace("_", " "),
        )
        axes[1].plot(wave, fit["standardized_residuals"], ".", label=name.replace("_", " "))
    axes[0].set(ylabel="Transit depth (ppm)", title="WASP-39b: conditional fixed-template check")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="grey", linewidth=0.7)
    axes[1].set(xlabel="Wavelength (micrometre)", ylabel="Residual / published error")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = run(args.seed)
    (args.output / "result.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    plot_result(output, args.output / "spectrum.png")
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in fit.items()
                    if k
                    in (
                        "chi_square_at_analytic_mean",
                        "degrees_of_freedom",
                        "mean_error_in_analytic_sigma",
                        "sampled_minus_analytic_log_evidence",
                        "posterior_predictive",
                    )
                }
                for name, fit in output["fits"].items()
            },
            indent=2,
        )
    )
