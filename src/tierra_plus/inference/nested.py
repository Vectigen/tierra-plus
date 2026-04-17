"""Baseline nested sampling surface for Tierra+."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

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

    nlive: int = 200
    dlogz: float = 0.1
    bound: str = "multi"
    sample: str = "rwalk"
    maxiter: int | None = None

    def __post_init__(self) -> None:
        if self.nlive <= 0:
            raise ValueError("NestedSamplerConfig.nlive must be positive.")
        if self.dlogz <= 0.0:
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
        )
        sampler.run_nested(
            dlogz=self.config.dlogz,
            maxiter=self.config.maxiter,
            print_progress=False,
        )
        return self._coerce_result(sampler.results)

    def _prior_transform(self, unit_cube: Sequence[float]) -> tuple[float, ...]:
        transformed = tuple(float(value) for value in self._model_bridge.prior_transform(unit_cube))
        if len(transformed) != len(self._parameter_names):
            raise ValueError(
                "NestedSampler model bridge prior_transform output must match parameter_names."
            )
        return transformed

    def _log_likelihood(self, parameters: Sequence[float]) -> float:
        parameter_tuple = tuple(float(value) for value in parameters)
        if len(parameter_tuple) != len(self._parameter_names):
            raise ValueError(
                "NestedSampler log-likelihood input dimensionality must match parameter_names."
            )
        return float(self._model_bridge.log_likelihood(parameter_tuple))

    def _coerce_result(self, payload: object) -> SamplerResult:
        try:
            result_payload = cast(NestedSamplerResults, payload)
            samples = _coerce_matrix(result_payload.samples)
            log_likelihoods = _coerce_vector(result_payload.logl)
            log_evidence = _coerce_scalar(result_payload.logz)
        except AttributeError as error:
            raise RuntimeError(
                "Nested sampler results must provide samples, logl, and logz attributes."
            ) from error

        if len(samples) != len(log_likelihoods):
            raise RuntimeError(
                "Nested sampler results must align samples with log-likelihood values."
            )

        metadata: dict[str, object] = {
            "nlive": self.config.nlive,
            "dlogz": self.config.dlogz,
            "bound": self.config.bound,
            "sample": self.config.sample,
        }
        for field_name in ("ncall", "niter"):
            if hasattr(payload, field_name):
                metadata[field_name] = int(getattr(payload, field_name))

        return SamplerResult(
            parameter_names=self._parameter_names,
            samples=samples,
            log_likelihoods=log_likelihoods,
            log_evidence=log_evidence,
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
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) == 0:
            raise RuntimeError("Nested sampler logz output must not be empty.")
        return float(values[-1])
    return float(cast(float, values))
