"""Baseline nested sampling surface for Tierra+."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from tierra_plus.exceptions import IntegrationNotConfiguredError

from .base import Sampler, SamplerResult


class ModelBridge(Protocol):
    """Likelihood adapter required by the baseline nested sampler."""

    parameter_names: tuple[str, ...]

    def prior_transform(self, unit_cube: Sequence[float]) -> Sequence[float]:
        """Map a unit-cube sample into the physical parameter space."""

    def log_likelihood(self, parameters: Sequence[float]) -> float:
        """Evaluate the log-likelihood for a physical parameter vector."""


class NestedSamplerHandle(Protocol):
    """Subset of the dynesty nested sampler API used by Tierra+."""

    results: object

    def run_nested(self, **kwargs: object) -> object:
        """Execute the nested sampler."""


class NestedSamplerResults(Protocol):
    """Subset of dynesty results consumed by Tierra+."""

    samples: object
    logl: object
    logz: object
    logwt: object
    logzerr: object


NestedSamplerFactory = Callable[..., NestedSamplerHandle]


@dataclass(frozen=True)
class NestedSamplerConfig:
    """Configuration for the baseline nested sampler.

    Parameters
    ----------
    nlive
        Number of live points maintained by the sampler.
    dlogz
        Evidence convergence threshold passed to ``dynesty``.
    bound
        Bounding strategy passed to ``dynesty``.
    sample
        Internal proposal strategy passed to ``dynesty``.
    maxiter
        Optional hard iteration cap.
    """

    seed: int = 0
    nlive: int = 200
    dlogz: float = 0.1
    bound: str = "multi"
    sample: str = "rwalk"
    maxiter: int | None = None

    def __post_init__(self) -> None:
        if self.nlive <= 0:
            raise ValueError("NestedSamplerConfig.nlive must be positive.")
        if not np.isfinite(self.dlogz) or self.dlogz <= 0.0:
            raise ValueError("NestedSamplerConfig.dlogz must be positive.")
        if not self.bound.strip():
            raise ValueError("NestedSamplerConfig.bound must not be empty.")
        if not self.sample.strip():
            raise ValueError("NestedSamplerConfig.sample must not be empty.")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("NestedSamplerConfig.maxiter must be positive when provided.")


class NestedSampler(Sampler):
    """Explicit nested-sampling lane backed by a model bridge."""

    def __init__(
        self,
        *,
        config: NestedSamplerConfig | None = None,
        model_bridge: ModelBridge | None,
        sampler_factory: NestedSamplerFactory | None = None,
    ) -> None:
        if model_bridge is None:
            raise IntegrationNotConfiguredError(
                "NestedSampler requires a model bridge with prior_transform "
                "and log_likelihood methods."
            )
        parameter_names = tuple(str(name) for name in model_bridge.parameter_names)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique")
        if not parameter_names:
            raise ValueError("NestedSampler model bridge must define at least one parameter name.")

        self.config = config or NestedSamplerConfig()
        self._model_bridge = model_bridge
        self._parameter_names = parameter_names
        self._sampler_factory = sampler_factory

    def run(self) -> SamplerResult:
        """Execute the configured nested sampler."""

        sampler_factory = self._sampler_factory or _load_dynesty_nested_sampler()
        sampler = sampler_factory(
            loglikelihood=self._log_likelihood,
            prior_transform=self._prior_transform,
            ndim=len(self._parameter_names),
            nlive=self.config.nlive,
            bound=self.config.bound,
            sample=self.config.sample,
            rstate=np.random.default_rng(self.config.seed),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            sampler.run_nested(
                dlogz=self.config.dlogz,
                maxiter=self.config.maxiter,
                print_progress=False,
            )
        result = self._coerce_result(sampler.results)
        return replace(
            result,
            metadata={
                **result.metadata,
                "warnings": [
                    {"category": item.category.__name__, "message": str(item.message)}
                    for item in captured
                ],
            },
        )

    def _prior_transform(self, unit_cube: Sequence[float]) -> tuple[float, ...]:
        cube = np.asarray(unit_cube, dtype=float)
        if (
            cube.shape != (len(self._parameter_names),)
            or not np.all(np.isfinite(cube))
            or np.any((cube < 0) | (cube > 1))
        ):
            raise ValueError("prior input must be a finite unit-cube vector")
        transformed = tuple(float(value) for value in self._model_bridge.prior_transform(unit_cube))
        if len(transformed) != len(self._parameter_names):
            raise ValueError(
                "NestedSampler model bridge prior_transform output must match parameter_names."
            )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("prior_transform must return finite values")
        return transformed

    def _log_likelihood(self, parameters: Sequence[float]) -> float:
        parameter_tuple = tuple(float(value) for value in parameters)
        if len(parameter_tuple) != len(self._parameter_names):
            raise ValueError(
                "NestedSampler log-likelihood input dimensionality must match parameter_names."
            )
        value = float(self._model_bridge.log_likelihood(parameter_tuple))
        if np.isnan(value) or value == np.inf:
            raise ValueError("log likelihood must be finite or negative infinity")
        return value

    def _coerce_result(self, payload: object) -> SamplerResult:
        try:
            result_payload = cast(NestedSamplerResults, payload)
            samples = _coerce_matrix(result_payload.samples)
            log_likelihoods = _coerce_vector(result_payload.logl)
            log_evidence = _coerce_scalar(result_payload.logz)
            log_evidence_error = _coerce_scalar(result_payload.logzerr)
            log_weights = np.asarray(result_payload.logwt, dtype=float)
            if (
                log_weights.shape != (len(samples),)
                or np.any(np.isnan(log_weights) | (log_weights == np.inf))
                or not np.any(np.isfinite(log_weights))
            ):
                raise RuntimeError("Nested sampler must return aligned finite log weights")
            unnormalized = np.exp(log_weights - np.max(log_weights))
            weights = tuple(float(w) for w in unnormalized / np.sum(unnormalized))
        except AttributeError as error:
            raise RuntimeError(
                "Nested sampler results must provide samples, logl, logz, "
                "logzerr, and logwt attributes."
            ) from error

        if not samples or any(len(row) != len(self._parameter_names) for row in samples):
            raise RuntimeError("Nested sampler sample dimensions must match parameter_names")
        if (
            not np.isfinite(log_evidence)
            or not np.isfinite(log_evidence_error)
            or log_evidence_error < 0
        ):
            raise RuntimeError("Nested sampler evidence must be finite")
        if not np.all(np.isfinite(samples)) or any(
            np.isnan(v) or v == np.inf for v in log_likelihoods
        ):
            raise RuntimeError("Nested sampler values must be finite (log likelihood may be -inf)")
        if len(samples) != len(log_likelihoods):
            raise RuntimeError(
                "Nested sampler results must align samples with log-likelihood values."
            )

        metadata: dict[str, object] = {
            "seed": self.config.seed,
            "nlive": self.config.nlive,
            "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
            "maxiter": self.config.maxiter,
            "iteration_cap_configured": self.config.maxiter is not None,
            "dlogz": self.config.dlogz,
            "bound": self.config.bound,
            "sample": self.config.sample,
        }
        for field_name in ("ncall", "niter"):
            if hasattr(payload, field_name):
                metadata[field_name] = int(np.asarray(getattr(payload, field_name)).sum())

        return SamplerResult(
            parameter_names=self._parameter_names,
            samples=samples,
            log_likelihoods=log_likelihoods,
            log_evidence=log_evidence,
            weights=weights,
            log_evidence_error=log_evidence_error,
            metadata=metadata,
        )


def _load_dynesty_nested_sampler() -> NestedSamplerFactory:
    try:
        dynesty = import_module("dynesty")
    except ModuleNotFoundError as error:
        raise IntegrationNotConfiguredError(
            "dynesty is required for NestedSampler.run(); install the "
            "inference dependency or provide sampler_factory explicitly."
        ) from error

    nested_sampler = getattr(dynesty, "NestedSampler", None)
    if nested_sampler is None:
        raise IntegrationNotConfiguredError(
            "dynesty is installed but does not expose NestedSampler."
        )
    return cast(NestedSamplerFactory, nested_sampler)


def _coerce_matrix(values: object) -> tuple[tuple[float, ...], ...]:
    rows = cast(Sequence[Sequence[float]], values)
    return tuple(tuple(float(entry) for entry in row) for row in rows)


def _coerce_vector(values: object) -> tuple[float, ...]:
    return tuple(float(entry) for entry in cast(Sequence[float], values))


def _coerce_scalar(values: object) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise RuntimeError("Nested sampler scalar output must not be empty.")
    return float(array.reshape(-1)[-1])
