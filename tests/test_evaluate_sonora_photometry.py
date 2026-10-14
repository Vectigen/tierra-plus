from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tierra_plus.diagnostics.physical_grid import FiniteSpectralGrid


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples/evaluate_sonora_photometry.py"
    spec = importlib.util.spec_from_file_location("evaluate_sonora_photometry", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Sonora evaluation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manual_offset_chi2(observation, model, sigma):
    residual = observation - model
    offset = np.sum(residual / sigma**2) / np.sum(1 / sigma**2)
    return np.sum(((residual - offset) / sigma) ** 2)


def test_checked_inference_matches_independent_offset_arithmetic() -> None:
    module = _load_module()
    spectra = np.array(
        [
            [[0.0, 1.0, 0.0], [0.2, 0.7, 0.1]],
            [[1.0, 0.0, 1.0], [0.8, 0.3, 0.9]],
        ]
    )
    labels = np.array([900.0, 1000.0])
    sigma = np.array([0.5, 1.0, 2.0])
    observation = np.array([[0.1, 0.8, 0.05]])
    result = module._infer_checked(
        FiniteSpectralGrid(spectra, labels, sigma), observation, "arithmetic_control"
    )

    chi2 = np.array(
        [
            _manual_offset_chi2(observation[0], model, sigma)
            for model in spectra.reshape(-1, spectra.shape[-1])
        ]
    )
    weights = np.exp(-0.5 * (chi2 - chi2.min()))
    posterior = (weights / weights.sum()).reshape(2, 2).sum(axis=1)
    np.testing.assert_allclose(result["chi2"], [chi2.min()], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["mean"], [posterior @ labels], rtol=1e-13, atol=1e-13)


def test_support_grid_has_declared_temperature_and_nuisance_axes() -> None:
    module = _load_module()
    rows = []
    for metallicity in (0.5, -0.5):
        for temperature in (1000.0, 900.0):
            for gravity in (316.0, 31.6):
                rows.append((temperature, gravity, metallicity))
    arrays = {
        "effective_temperature_k": np.array([row[0] for row in rows]),
        "gravity_m_s2": np.array([row[1] for row in rows]),
        "metallicity_dex": np.array([row[2] for row in rows]),
        "fsed": np.zeros(len(rows), dtype=int),
        "magnitudes": np.array(
            [[row[0] / 100 + row[1] / 100 + row[2] + band for band in range(3)] for row in rows]
        ),
    }
    config = {
        "expected_grid": {
            "temperatures_k": [900.0, 1000.0],
            "metallicity_dex": [-0.5, 0.5],
        },
        "partition": {"support": {"gravity_m_s2": [31.6, 316.0]}},
    }
    grid = module.build_support_grid(arrays, config, np.ones(3))

    assert grid.spectra.shape == (2, 4, 3)
    np.testing.assert_array_equal(grid.labels, [900.0, 1000.0])
    # Nuisance order is gravity-major, then metallicity as declared in the protocol.
    np.testing.assert_allclose(grid.spectra[0, 0], [8.816, 9.816, 10.816])
    np.testing.assert_allclose(grid.spectra[1, 3], [13.66, 14.66, 15.66])


def test_selection_uses_only_frozen_score_and_width_rules() -> None:
    module = _load_module()
    result = {
        "mean": np.array([900.0, 1050.0, 1100.0, 1200.0]),
        "lower": np.array([900.0, 900.0, 1000.0, 1100.0]),
        "upper": np.array([900.0, 1200.0, 1400.0, 1200.0]),
        "envelope_lower": np.array([900.0, 900.0, 900.0, 1000.0]),
        "envelope_upper": np.array([1000.0, 1300.0, 1400.0, 1300.0]),
        "chi2": np.array([1.0, 2.0, 1.0, 3.0]),
    }
    truth = np.array([900.0, 1000.0, 1100.0, 1200.0])
    metrics = module.selection_metrics(result, truth, threshold=2.0, width_cap=300.0)

    # The support-boundary interval is accepted because no boundary rule was frozen.
    assert metrics["accepted"] == 2
    assert metrics["score_accepted"] == 3
    assert metrics["width_accepted"] == 3
    assert metrics["accepted_covered"] == 2
    assert metrics["acceptance"] == 0.5
    assert metrics["accepted_coverage"] == 1.0


def test_nonfinite_inference_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    grid = FiniteSpectralGrid(np.zeros((2, 1, 3)), np.array([900.0, 1000.0]), np.ones(3))

    def invalid(_grid, _observations):
        values = np.array([np.nan])
        return {
            "mean": values,
            "lower": values,
            "upper": values,
            "envelope_lower": values,
            "envelope_upper": values,
            "chi2": values,
        }

    monkeypatch.setattr(FiniteSpectralGrid, "infer", invalid)
    with pytest.raises(ValueError, match="non-finite"):
        module._infer_checked(grid, np.zeros((1, 3)), "invalid_control")


def test_gate_failure_reasons_are_exact() -> None:
    module = _load_module()
    planned = {"minimum_acceptance": 0.7, "minimum_accepted_coverage": 0.85}
    assert module._gate_failures(
        {"acceptance": 0.69, "accepted_coverage": 0.84}, planned, "validation"
    ) == [
        "validation_acceptance_below_floor",
        "validation_accepted_coverage_below_floor",
    ]
    assert (
        module._gate_failures({"acceptance": 0.7, "accepted_coverage": 0.85}, planned, "validation")
        == []
    )


def test_validation_failure_does_not_infer_cloud_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    arrays = {
        "magnitudes": np.zeros((4, 2)),
        "effective_temperature_k": np.array([900.0, 1000.0, 1100.0, 1200.0]),
        "calibration_indices": np.array([0]),
        "validation_indices": np.array([1]),
        "cloud_challenge_indices": np.array([2, 3]),
    }
    config = {
        "question": "control",
        "claim_boundary": "control only",
        "planned_evaluation": {
            "status": "not_run",
            "noise_seed": [20261014, 2],
            "independent_sigma_magnitude": 0.05,
            "score_quantile": 0.9,
            "minimum_acceptance": 0.7,
            "minimum_accepted_coverage": 0.85,
            "maximum_temperature_interval_width_k": 300,
            "likelihood": "control",
        },
    }
    roles: list[str] = []

    monkeypatch.setattr(module, "build_support_grid", lambda *_args: object())

    def infer(_grid, observations, role):
        roles.append(role)
        if role == "cloud_challenge":
            raise AssertionError("challenge must not be inferred after validation failure")
        score = 1.0 if role == "calibration" else 2.0
        size = len(observations)
        return {
            "mean": np.full(size, 1000.0),
            "lower": np.full(size, 900.0),
            "upper": np.full(size, 1000.0),
            "envelope_lower": np.full(size, 900.0),
            "envelope_upper": np.full(size, 1000.0),
            "chi2": np.full(size, score),
        }

    monkeypatch.setattr(module, "_infer_checked", infer)
    report = module.evaluate(arrays, config, {})

    assert roles == ["calibration", "validation"]
    assert report["verdict"] == "validation_failed"
    assert report["cloud_challenge"] == {"status": "not_run_validation_failed"}


def test_report_writer_refuses_to_replace_result(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "record.json"
    module.write_report({"verdict": "control"}, output)
    with pytest.raises(ValueError, match="Refusing to replace"):
        module.write_report({"verdict": "changed"}, output)
