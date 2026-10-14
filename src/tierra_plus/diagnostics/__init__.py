"""Analytic controls for falsifiable spectral inference experiments."""

from .linear import (
    constrained_linear_estimator,
    gaussian_posterior,
    gaussian_posterior_operator,
    nuisance_projected_gls,
    spectral_design,
)

__all__ = [
    "constrained_linear_estimator",
    "gaussian_posterior",
    "gaussian_posterior_operator",
    "nuisance_projected_gls",
    "spectral_design",
]
