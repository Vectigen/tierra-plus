"""Exact finite-grid spectral adequacy checks, conditional on the supplied models.

This module detects neither omitted physics nor parameter error by construction.
Its scores require empirical validation. The constant-depth offset is integrated
under a common flat prior; withheld bins never enter their own posterior fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tierra_plus.diagnostics.physical_grid import _finite, remove_offset

FloatArray = NDArray[np.float64]


def _logsumexp(x: FloatArray) -> FloatArray:
    maximum = x.max(axis=1)
    result: FloatArray = maximum + np.log(np.exp(x - maximum[:, None]).sum(axis=1))
    return result


@dataclass(frozen=True)
class SpectralAdequacy:
    """Finite candidate spectra with explicit prior and equal-count spectral blocks.

    Spectra are (candidate, wavelength), sorted in increasing wavelength by the
    caller. Prior mass is per candidate: a union of chemical families must set
    equal *family* mass explicitly when their grid sizes differ. Intervals are
    discrete-label central intervals, not continuous atmospheric uncertainties.
    """

    spectra: FloatArray
    labels: FloatArray
    sigma: FloatArray
    prior: FloatArray | None = None
    blocks: int = 4

    def __post_init__(self) -> None:
        spectra = _finite(self.spectra, 2, "spectra").copy()
        labels = _finite(self.labels, 1, "labels").copy()
        sigma = _finite(self.sigma, 1, "sigma").copy()
        if spectra.shape != (labels.size, sigma.size) or np.any(sigma <= 0):
            raise ValueError("Matching candidate labels and positive bin errors are required")
        if type(self.blocks) is not int or not 2 <= self.blocks <= sigma.size:
            raise ValueError("blocks must be an integer between two and the bin count")
        prior = (
            np.ones(labels.size, dtype=np.float64)
            if self.prior is None
            else _finite(self.prior, 1, "prior").copy()
        )
        if prior.shape != labels.shape or np.any(prior <= 0) or not np.isfinite(prior.sum()):
            raise ValueError("Each candidate needs positive finite prior mass")
        prior /= prior.sum()
        for name, value in (
            ("spectra", spectra),
            ("labels", labels),
            ("sigma", sigma),
            ("prior", prior),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def evaluate(
        self, observations: ArrayLike, chunk_size: int = 128, predictive: bool = True
    ) -> dict[str, FloatArray]:
        """Return posterior summaries, cheap controls and withheld-block surprise.

        ``predictive_blocks`` is (-2 log p(y_B|y_A) minus the independent-noise
        Gaussian normalizing constant), divided by the withheld bin count.
        It includes the rank-one offset posterior covariance. Equivalently,
        it is [2(log Z_A - log Z_full) + log(P_full/P_A)] / |B|, where each
        Z sums prior-weighted exp(-chi²/2) and P is offset precision sum(1/sigma²).
        ``predictive_max`` is the fixed maximum over blocks, to calibrate once.

        ``fitted_block_max`` instead evaluates those blocks at the full-spectrum
        minimum-chi² candidate and full-fit offset. This deliberately cheaper
        comparator reuses the withheld bins during fitting.
        """
        data = _finite(observations, 2, "observations")
        if data.shape[1] != self.sigma.size:
            raise ValueError("Observation and grid wavelength counts differ")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        assert self.prior is not None
        log_prior = np.log(self.prior)
        bins = self.sigma.size
        partitions = np.array_split(np.arange(bins), self.blocks)
        # Removing a common spectral anchor improves distance accuracy without
        # changing residuals, priors or the offset integral.
        anchor = self.spectra[0]
        models = self.spectra - anchor
        full_models = remove_offset(models, self.sigma)
        full_norm = np.sum(full_models**2, axis=1)
        complement_models, complement_norms, complements = [], [], []
        for block in partitions:
            keep = np.ones(bins, dtype=bool)
            keep[block] = False
            complements.append(keep)
            projected = remove_offset(models[:, keep], self.sigma[keep])
            complement_models.append(projected)
            complement_norms.append(np.sum(projected**2, axis=1))
        unique_labels, inverse = np.unique(self.labels, return_inverse=True)
        membership = np.eye(len(unique_labels))[inverse]
        keys = ["mean", "lower", "upper", "entropy", "chi2", "fitted_block_max"]
        if predictive:
            keys += ["predictive_blocks", "predictive_max"]
        chunks: dict[str, list[FloatArray]] = {key: [] for key in keys}
        for start in range(0, len(data), chunk_size):
            raw = data[start : start + chunk_size] - anchor
            y = remove_offset(raw, self.sigma)
            relative = y @ full_models.T - full_norm / 2
            logits = relative - relative.max(axis=1, keepdims=True) + log_prior
            log_z_full_centered = _logsumexp(logits)
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = (weights / weights.sum(axis=1, keepdims=True)) @ membership
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative[:, -1] = 1
            chunks["mean"].append(probabilities @ unique_labels)
            chunks["lower"].append(unique_labels[np.argmax(cumulative >= 0.05, axis=1)])
            chunks["upper"].append(unique_labels[np.argmax(cumulative >= 0.95, axis=1)])
            chunks["entropy"].append(
                -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
            )
            best = np.argmax(relative, axis=1)
            residual = y - full_models[best]
            minimum_chi2 = np.sum(residual**2, axis=1)
            chunks["chi2"].append(minimum_chi2)
            chunks["fitted_block_max"].append(
                np.max([np.mean(residual[:, block] ** 2, axis=1) for block in partitions], axis=0)
            )
            if predictive:
                block_scores = []
                precision = np.sum(1 / self.sigma**2)
                for block, keep, projected, norm in zip(
                    partitions, complements, complement_models, complement_norms, strict=True
                ):
                    y_a = remove_offset(raw[:, keep], self.sigma[keep])
                    relative_a = y_a @ projected.T - norm / 2
                    centered_a = relative_a - relative_a.max(axis=1, keepdims=True)
                    log_z_a_centered = _logsumexp(centered_a + log_prior)
                    residual_a = y_a - projected[np.argmax(relative_a, axis=1)]
                    minimum_chi2_a = np.sum(residual_a**2, axis=1)
                    score = (
                        minimum_chi2
                        - minimum_chi2_a
                        + 2 * (log_z_a_centered - log_z_full_centered)
                        + np.log(precision / np.sum(1 / self.sigma[keep] ** 2))
                    ) / len(block)
                    block_scores.append(score)
                matrix = np.stack(block_scores, axis=1)
                chunks["predictive_blocks"].append(matrix)
                chunks["predictive_max"].append(matrix.max(axis=1))
        result = {key: np.concatenate(value) for key, value in chunks.items()}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Spectral adequacy calculation produced nonfinite results")
        return result
