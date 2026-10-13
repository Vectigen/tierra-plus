from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tierra_plus.contamination import contaminate_transit
from tierra_plus.forward.opacity import get_opacity_model
from tierra_plus.forward.opacity.ensemble import load_benchmark_case, posterior_weights_for_case
from tierra_plus.inference import NestedSampler, NestedSamplerConfig
from tierra_plus.inference.spectral import SpectralLikelihood, UniformPrior
from tierra_plus.models import Spectrum


@pytest.mark.parametrize("values", [(np.nan, 1), (np.inf, 1)])
def test_nonfinite_data_rejected(values: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        Spectrum([1, 2], list(values), [0.1, 0.1])


def test_manifest_cannot_escape_root(tmp_path: Path) -> None:
    asset = get_opacity_model("synthetic-opacity-reference")
    with pytest.raises(ValueError, match="inside"):
        replace(asset, relative_path=Path("../../secret")).resolve_path(tmp_path)
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        replace(asset, relative_path=Path("linked/secret")).resolve_path(tmp_path)


def test_contamination_limits_and_uncertainties() -> None:
    planet = Spectrum([1, 2], [0.01, 0.02], [0.001, 0.002])
    assert contaminate_transit(planet, [2, 2], [1, 1], [3, 3]) == planet
    contaminated = contaminate_transit(planet, [2, 2], [1, 1], [3, 3], 0.2, 0)
    assert np.allclose(contaminated.flux, np.asarray(planet.flux) / 0.9)
    assert np.allclose(contaminated.uncertainty, np.asarray(planet.uncertainty) / 0.9)
    with pytest.raises(ValueError, match="fractions"):
        contaminate_transit(planet, [2, 2], [1, 1], [3, 3], 0.6, 0.6)
    with pytest.raises(ValueError, match="positive"):
        contaminate_transit(planet, [0, 2], [1, 1], [3, 3])


def test_fixed_model_priors_validated() -> None:
    case = load_benchmark_case("synthetic-opacity-reference")
    with pytest.raises(ValueError, match="Priors"):
        posterior_weights_for_case(case, {})
    with pytest.raises(ValueError, match="positive"):
        posterior_weights_for_case(case, dict.fromkeys(case.candidates, np.nan))


def test_real_dynesty_matches_gaussian_posterior_and_evidence() -> None:
    pytest.importorskip("dynesty")
    observed = Spectrum([1.0], [0.3], [0.1])
    bridge = SpectralLikelihood(
        observed, lambda theta: Spectrum([1.0], [theta[0]]), [UniformPrior("mean", -2, 2)]
    )
    result = NestedSampler(
        config=NestedSamplerConfig(nlive=150, dlogz=0.05, seed=12), model_bridge=bridge
    ).run()
    samples, weights = np.asarray(result.samples)[:, 0], np.asarray(result.weights)
    mean = np.sum(samples * weights)
    deviation = np.sqrt(np.sum(weights * (samples - mean) ** 2))
    assert weights.sum() == pytest.approx(1.0)
    assert mean == pytest.approx(0.3, abs=0.03)
    assert deviation == pytest.approx(0.1, abs=0.025)
    # Integral of this normalized Gaussian over [-2,2] divided by prior width.
    assert result.log_evidence == pytest.approx(-np.log(4), abs=0.45)
    assert result.log_evidence_error > 0
    assert result.metadata["ncall"] > 0


def test_numpy_result_coercion_and_missing_weights() -> None:
    bridge = SpectralLikelihood(
        Spectrum([1], [0], [1]), lambda p: Spectrum([1], [p[0]]), [UniformPrior("x", -1, 1)]
    )
    sampler = NestedSampler(model_bridge=bridge)
    payload = SimpleNamespace(
        samples=np.array([[0.0], [0.1]]),
        logl=np.array([-1.0, -1.1]),
        logz=np.array([-3.0, -2.0]),
        logwt=np.log([0.2, 0.8]),
        logzerr=np.array([0.2, 0.1]),
        ncall=np.array([2, 3]),
    )
    result = sampler._coerce_result(payload)
    assert result.weights == pytest.approx((0.2, 0.8))
    assert result.metadata["ncall"] == 5
    del payload.logwt
    with pytest.raises(RuntimeError, match="logwt"):
        sampler._coerce_result(payload)
