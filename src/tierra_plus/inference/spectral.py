"""Validated spectrum-to-sampler bridge with explicit independent uniform priors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tierra_plus.forward.opacity.ensemble import gaussian_log_likelihood
from tierra_plus.models import Spectrum


@dataclass(frozen=True)
class UniformPrior:
    """One finite, bounded physical parameter prior."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or not np.isfinite([self.lower, self.upper]).all()
            or self.lower >= self.upper
        ):
            raise ValueError("Uniform priors need a name and finite ordered bounds")


class SpectralLikelihood:
    """Bridge a callable spectrum model and diagonal Gaussian data to nested sampling.

    Parameters
    ----------
    observed
        Data including positive uncertainties.
    model
        Callable mapping an ordered parameter vector to a spectrum on the data grid.
    priors
        Independent uniform priors, in the order expected by ``model``.
    """

    def __init__(
        self,
        observed: Spectrum,
        model: Callable[[Sequence[float]], Spectrum],
        priors: Sequence[UniformPrior],
    ) -> None:
        self.observed, self.model, self.priors = observed, model, tuple(priors)
        self.parameter_names = tuple(prior.name for prior in priors)
        if observed.uncertainty is None:
            raise ValueError("Retrieval data require uncertainties")
        if not self.priors or len(set(self.parameter_names)) != len(self.priors):
            raise ValueError("Priors must have unique parameter names")

    def prior_transform(self, unit_cube: Sequence[float]) -> tuple[float, ...]:
        """Transform a finite unit-cube vector into the configured bounds."""
        cube = np.asarray(unit_cube, dtype=float)
        if (
            cube.shape != (len(self.priors),)
            or not np.isfinite(cube).all()
            or np.any((cube < 0) | (cube > 1))
        ):
            raise ValueError("Expected one finite unit-cube coordinate per prior")
        return tuple(
            p.lower + float(u) * (p.upper - p.lower) for p, u in zip(self.priors, cube, strict=True)
        )

    def log_likelihood(self, parameters: Sequence[float]) -> float:
        """Evaluate the fully normalized Gaussian likelihood on matching grids."""
        if len(parameters) != len(self.priors) or not np.isfinite(parameters).all():
            raise ValueError("Parameter vector must be finite and match the priors")
        if any(v < p.lower or v > p.upper for p, v in zip(self.priors, parameters, strict=True)):
            return -np.inf
        return gaussian_log_likelihood(self.observed, self.model(parameters))
