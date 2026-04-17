from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import tierra_plus.inference.nested as nested_module
from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.inference import NestedSampler, NestedSamplerConfig, Sampler, SamplerResult


@dataclass(frozen=True)
class DummyBridge:
    parameter_names: tuple[str, ...] = ("temperature_kelvin", "log_h2o")

    def prior_transform(self, unit_cube: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in unit_cube)

    def log_likelihood(self, parameters: Sequence[float]) -> float:
        return -sum(float(value) ** 2 for value in parameters)


class FakeDynestySampler:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.results = SimpleNamespace(
            samples=((0.2, 0.4), (0.6, 0.8)),
            logl=(-0.2, -1.0),
            logz=(-3.0, -2.5),
            ncall=24,
            niter=12,
        )
        self.run_kwargs: dict[str, object] | None = None

    def run_nested(self, **kwargs: object) -> None:
        self.run_kwargs = kwargs


def test_nested_sampler_implements_sampler_interface() -> None:
    created: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeDynestySampler:
        created.update(kwargs)
        return FakeDynestySampler(**kwargs)

    sampler = NestedSampler(
        config=NestedSamplerConfig(nlive=32, dlogz=0.5, sample="rwalk", bound="multi"),
        model_bridge=DummyBridge(),
        sampler_factory=factory,
    )

    assert isinstance(sampler, Sampler)

    result = sampler.run()

    assert isinstance(result, SamplerResult)
    assert result.parameter_names == ("temperature_kelvin", "log_h2o")
    assert result.samples == ((0.2, 0.4), (0.6, 0.8))
    assert result.log_likelihoods == (-0.2, -1.0)
    assert result.log_evidence == -2.5
    assert result.metadata["nlive"] == 32
    assert created["ndim"] == 2
    assert callable(created["prior_transform"])
    assert callable(created["loglikelihood"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nlive": 0}, "nlive"),
        ({"dlogz": 0.0}, "dlogz"),
        ({"sample": ""}, "sample"),
        ({"bound": ""}, "bound"),
        ({"maxiter": 0}, "maxiter"),
    ],
)
def test_nested_sampler_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NestedSamplerConfig(**kwargs)


def test_nested_sampler_requires_model_bridge() -> None:
    with pytest.raises(IntegrationNotConfiguredError, match="model bridge"):
        NestedSampler(config=NestedSamplerConfig(), model_bridge=None)


def test_nested_sampler_requires_dynesty_when_no_factory_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dynesty(_: str) -> object:
        raise ModuleNotFoundError("dynesty")

    monkeypatch.setattr(nested_module, "import_module", missing_dynesty)

    sampler = NestedSampler(config=NestedSamplerConfig(), model_bridge=DummyBridge())

    with pytest.raises(IntegrationNotConfiguredError, match="dynesty"):
        sampler.run()
