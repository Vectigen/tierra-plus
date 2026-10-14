"""Exact linear controls for a whitened, two-component spectral experiment.

These functions define a controlled statistical problem, not a radiative-transfer
model. Observations are centered and divided by their known noise standard
deviation before entering the posterior and generalized least-squares routines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _finite_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _design_array(design: ArrayLike) -> FloatArray:
    array = _finite_array(design, "design")
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 2:
        raise ValueError("design must have shape (n, 2), with n >= 1")
    return array


def _observations_array(observations: ArrayLike, n: int) -> FloatArray:
    array = _finite_array(observations, "observations")
    if array.ndim not in (1, 2) or array.shape[-1] != n:
        raise ValueError("observations must have shape (n,) or (batch, n)")
    if array.ndim == 2 and array.shape[0] == 0:
        raise ValueError("observations must contain at least one observation")
    return array


def spectral_design(
    template: Sequence[float] | FloatArray,
    sigma: Sequence[float] | FloatArray,
    overlap: float,
    target_snr: float = 2.0,
    nuisance_snr: float = 6.0,
) -> FloatArray:
    """Construct two whitened spectral directions with known cosine overlap.

    The target direction is ``template / sigma`` with its projection onto
    ``1 / sigma`` removed. After unit normalization, the nuisance direction is
    ``overlap * target + sqrt(1 - overlap**2) * normalized(1 / sigma)``.
    Column norms are ``target_snr`` and ``nuisance_snr``. Thus these values are
    signal-to-noise norms for unit latent amplitudes, not per-bin errors.

    ``overlap`` may include -1 and 1: these produce an intentionally rank-one
    design. A constant template cannot define a target and is rejected.
    """
    spectrum = _finite_array(template, "template")
    errors = _finite_array(sigma, "sigma")
    if spectrum.ndim != 1 or len(spectrum) < 2 or errors.shape != spectrum.shape:
        raise ValueError("template and sigma must be matching vectors with at least two bins")
    if np.any(errors <= 0):
        raise ValueError("sigma must be strictly positive")
    if not np.isfinite(overlap) or not -1 <= overlap <= 1:
        raise ValueError("overlap must be finite and in [-1, 1]")
    if not np.isfinite([target_snr, nuisance_snr]).all() or min(target_snr, nuisance_snr) <= 0:
        raise ValueError("signal-to-noise norms must be finite and strictly positive")

    # Scaling the offset before normalization avoids squaring very large inverse
    # errors. Center in depth units first to avoid subtracting a large continuum
    # after whitening. Both operations preserve the requested directions.
    offset = errors.min() / errors
    offset /= np.linalg.norm(offset)
    anchor = spectrum[0]
    centered = spectrum - anchor
    weighted_offset = float(np.dot(offset**2, centered))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        signal = (centered - weighted_offset) / errors
    if not np.isfinite(signal).all():
        raise ValueError("template and sigma scales cannot be safely whitened")
    # A final projection removes accumulated floating-point roundoff.
    signal -= offset * np.dot(offset, signal)
    scale = float(np.max(np.abs(signal)))
    if scale == 0:
        raise ValueError("template must be nonconstant after removal of the depth offset")
    signal /= scale
    signal /= np.linalg.norm(signal)
    nuisance = overlap * signal + np.sqrt((1 - overlap) * (1 + overlap)) * offset
    return np.column_stack((target_snr * signal, nuisance_snr * nuisance))


def gaussian_posterior_operator(
    design: ArrayLike, *, rho: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Return posterior mean operator and covariance under an explicit prior.

    For ``theta ~ N(0, [[1, rho], [rho, 1]])`` and
    ``x | theta ~ N(design @ theta, I)``, the posterior mean is ``gain @ x``.
    ``rho`` specifies the assumed prior, which must be chosen independently of
    any benchmark labels. In particular, training and deployment priors are
    distinct baselines when their correlations differ.

    The two-by-two solve is performed in prior-whitened coordinates. It remains
    well-defined for rank-deficient or zero designs when ``abs(rho) < 1``.
    """
    matrix = _design_array(design)
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    prior = np.array([[1.0, rho], [rho, 1.0]])
    factor = np.linalg.cholesky(prior)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = matrix @ factor
        precision = np.eye(2) + transformed.T @ transformed
    if not np.isfinite(precision).all():
        raise ValueError("design scale is too large for finite posterior precision")
    covariance = factor @ np.linalg.solve(precision, factor.T)
    covariance = (covariance + covariance.T) / 2
    gain = factor @ np.linalg.solve(precision, transformed.T)
    return gain, covariance


def gaussian_posterior(
    design: ArrayLike, observations: ArrayLike, *, rho: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Condition the explicit Gaussian prior on one observation or a batch.

    Input shape ``(n,)`` gives a mean of shape ``(2,)``; ``(batch, n)`` gives
    ``(batch, 2)``. The returned shared posterior covariance has shape ``(2, 2)``.
    """
    matrix = _design_array(design)
    values = _observations_array(observations, len(matrix))
    gain, covariance = gaussian_posterior_operator(matrix, rho=rho)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = values @ gain.T
    if not np.isfinite(mean).all():
        raise ValueError("observation scale does not give finite posterior means")
    return mean, covariance


def constrained_linear_estimator(
    design: ArrayLike, rho: float, max_nuisance_response: float
) -> tuple[FloatArray, float]:
    """Find the minimum-MSE linear readout at a specified nuisance sensitivity.

    Minimize ``E[(w @ x - theta[0])**2]`` subject to
    ``abs(w @ design[:, 1]) <= max_nuisance_response`` under the same explicit
    Gaussian prior and identity-noise model as :func:`gaussian_posterior`.
    Return the readout vector and its exact expected mean squared error.

    This is a constrained optimum over linear readouts, not a universal bound
    on nonlinear predictors. It gives an analytic reliability/accuracy frontier
    without selecting a favorable accuracy threshold after observing a model.
    """
    matrix = _design_array(design)
    if not np.isfinite(max_nuisance_response) or max_nuisance_response < 0:
        raise ValueError("max_nuisance_response must be finite and nonnegative")
    gain, covariance = gaussian_posterior_operator(matrix, rho=rho)
    nuisance = matrix[:, 1]
    if not np.any(nuisance):
        raise ValueError("the nuisance column must be nonzero")
    readout = gain[0].copy()
    response = float(readout @ nuisance)
    target = float(np.clip(response, -max_nuisance_response, max_nuisance_response))
    if target != response:
        # Woodbury applies the inverse predictive covariance without constructing
        # an n-by-n matrix. The multiplier enforces the active scalar constraint.
        inverse_covariance_nuisance = nuisance - matrix @ (covariance @ (matrix.T @ nuisance))
        denominator = float(nuisance @ inverse_covariance_nuisance)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("input scales do not give positive finite constraint information")
        readout -= (response - target) / denominator * inverse_covariance_nuisance
    error = readout @ matrix - np.array([1.0, 0.0])
    prior = np.array([[1.0, rho], [rho, 1.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        expected_mse = float(error @ prior @ error + readout @ readout)
    if not np.isfinite(readout).all() or not np.isfinite(expected_mse):
        raise ValueError("input scales do not give a finite readout and expected error")
    return readout, expected_mse


def nuisance_projected_gls(
    design: ArrayLike, observations: ArrayLike
) -> tuple[float | FloatArray, float]:
    """Estimate the target while projecting out the nuisance, without a prior.

    Both columns must be identifiable. The returned variance is the exact
    sampling variance under the declared identity noise covariance. Its
    divergence near collinearity is real information loss; exact numerical rank
    degeneracy raises rather than manufacturing a point estimate.
    """
    matrix = _design_array(design)
    values = _observations_array(observations, len(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    if len(singular_values) < 2 or singular_values[-1] <= tolerance:
        raise ValueError("target and nuisance are not identifiable: design is rank deficient")
    nuisance = matrix[:, 1] / np.linalg.norm(matrix[:, 1])
    residual = matrix[:, 0] - nuisance * np.dot(nuisance, matrix[:, 0])
    information = float(np.dot(residual, residual))
    if not np.isfinite(information) or information <= 0:
        raise ValueError("design scale does not give finite positive target information")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        estimate = values @ (residual / information)
        variance = float(np.divide(1.0, information))
    if not np.isfinite(estimate).all() or not np.isfinite(variance):
        raise ValueError("input scales do not give a finite estimate and variance")
    return (float(estimate) if estimate.ndim == 0 else estimate), variance
