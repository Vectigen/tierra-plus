"""Finite spectral-grid controls with an analytically removed depth offset.

These are discrete-model calculations, not continuous atmospheric retrieval.
All likelihoods assume independent, known Gaussian measurement errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _finite(value: ArrayLike, ndim: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def shifted_bin_integrals(
    wavelength: ArrayLike, depth: ArrayLike, edges: ArrayLike, shifts: ArrayLike
) -> FloatArray:
    """Average a piecewise-linear spectrum at edges+shift, without extrapolation.

    The integral is analytic within each interpolation segment. A positive
    shift samples the supplied spectrum at longer wavelengths. Output axes
    are (shift, bin); this is a top-hat response, without an instrument LSF.
    """
    wave = _finite(wavelength, 1, "wavelength")
    flux = _finite(depth, 1, "depth")
    bounds = _finite(edges, 1, "edges")
    offsets = _finite(shifts, 1, "shifts")
    if wave.shape != flux.shape or len(wave) < 2 or len(bounds) < 2:
        raise ValueError("Matching spectra and at least two edges are required")
    if np.any(np.diff(wave) <= 0) or np.any(np.diff(bounds) <= 0):
        raise ValueError("Wavelengths and edges must be strictly increasing")
    points = bounds[None, :] + offsets[:, None]
    if points.min() < wave[0] or points.max() > wave[-1]:
        raise ValueError("Shifted bins require extrapolation")
    widths = np.diff(wave)
    slopes = np.diff(flux) / widths
    primitive = np.concatenate(([0.0], np.cumsum(widths * (flux[:-1] + flux[1:]) / 2)))
    indices = np.clip(np.searchsorted(wave, points, side="right") - 1, 0, len(wave) - 2)
    dx = points - wave[indices]
    values = primitive[indices] + flux[indices] * dx + 0.5 * slopes[indices] * dx**2
    result: FloatArray = np.diff(values, axis=1) / np.diff(bounds)
    return result


def remove_offset(spectra: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """Whiten and project out a free constant-depth offset, exactly."""
    values = _finite(spectra, 2, "spectra")
    errors = _finite(sigma, 1, "sigma")
    if values.shape[1] != errors.size or np.any(errors <= 0):
        raise ValueError("Each bin requires a positive error")
    unit = 1 / errors
    unit = unit / np.linalg.norm(unit)
    scaled = values / errors
    return scaled - (scaled @ unit)[:, None] * unit


def calibration_radius(errors: ArrayLike, probability: float = 0.9) -> float:
    """Conservative split-conformal absolute-error order statistic.

    Exchangeability is required for a coverage guarantee. A heldout physical
    grid or changed nuisance distribution need not satisfy that assumption.
    """
    values = np.abs(_finite(errors, 1, "errors"))
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    rank = int(np.ceil((values.size + 1) * probability))
    if rank > values.size:
        raise ValueError("Too few calibration samples for a finite radius")
    return float(np.partition(values, rank - 1)[rank - 1])


def _summarize(probabilities: FloatArray, labels: FloatArray) -> dict[str, FloatArray]:
    order = np.argsort(labels, kind="stable")
    ordered = labels[order]
    cumulative = np.cumsum(probabilities[:, order], axis=1)
    # Floating point summation must not select beyond the last candidate.
    cumulative[:, -1] = 1.0
    return {
        "mean": probabilities @ labels,
        "lower": ordered[np.argmax(cumulative >= 0.05, axis=1)],
        "upper": ordered[np.argmax(cumulative >= 0.95, axis=1)],
    }


@dataclass(frozen=True)
class FiniteSpectralGrid:
    """Uniform-prior finite-grid posterior over physical state and nuisance.

    ``spectra`` has axes (physical state, nuisance state, wavelength). The
    common constant-offset prior integrates out with a candidate-independent
    factor. Exact means exact enumeration of this finite support only.
    """

    spectra: FloatArray
    labels: FloatArray
    sigma: FloatArray

    def __post_init__(self) -> None:
        spectra = _finite(self.spectra, 3, "spectra").copy()
        labels = _finite(self.labels, 1, "labels").copy()
        sigma = _finite(self.sigma, 1, "sigma").copy()
        if spectra.shape[0] != labels.size or spectra.shape[2] != sigma.size:
            raise ValueError("Grid, labels and errors have inconsistent shapes")
        if np.any(sigma <= 0) or spectra.shape[2] < 2:
            raise ValueError("At least two bins with positive errors are required")
        for name, value in (("spectra", spectra), ("labels", labels), ("sigma", sigma)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def infer(self, observations: ArrayLike, chunk_size: int = 128) -> dict[str, FloatArray]:
        """Return marginal estimates, conditional interval envelope, and min chi².

        An envelope contains every nuisance-conditional central interval.
        It has no generic frequentist coverage guarantee; the experiment must
        measure its coverage, width and selective acceptance together.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        data = remove_offset(observations, self.sigma)
        states, nuisances, bins = self.spectra.shape
        models = remove_offset(self.spectra.reshape(-1, bins), self.sigma)
        norms = np.sum(models**2, axis=1)
        chunks: dict[str, list[FloatArray]] = {
            key: []
            for key in ("mean", "lower", "upper", "envelope_lower", "envelope_upper", "chi2")
        }
        for start in range(0, len(data), chunk_size):
            y = data[start : start + chunk_size]
            logits = y @ models.T - norms / 2
            likelihood = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = likelihood / likelihood.sum(axis=1, keepdims=True)
            posterior = probabilities.reshape(-1, states, nuisances).sum(axis=2)
            summary = _summarize(posterior, self.labels)
            for key, value in summary.items():
                chunks[key].append(value)
            conditional_logits = logits.reshape(-1, states, nuisances)
            conditional_likelihood = np.exp(
                conditional_logits - conditional_logits.max(axis=1, keepdims=True)
            )
            conditional = conditional_likelihood / conditional_likelihood.sum(axis=1, keepdims=True)
            intervals = [
                _summarize(conditional[:, :, index], self.labels) for index in range(nuisances)
            ]
            chunks["envelope_lower"].append(np.min([x["lower"] for x in intervals], axis=0))
            chunks["envelope_upper"].append(np.max([x["upper"] for x in intervals], axis=0))
            # Cancellation can leave a tiny negative value for exact spectra.
            chunks["chi2"].append(np.maximum(0, np.sum(y**2, axis=1) - 2 * logits.max(axis=1)))
        return {key: np.concatenate(value) for key, value in chunks.items()}
