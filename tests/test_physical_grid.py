"""Scientific controls for the finite physical-grid experiment."""

import numpy as np
import pytest

from tierra_plus.diagnostics.physical_grid import (
    FiniteSpectralGrid,
    calibration_radius,
    remove_offset,
    shifted_bin_integrals,
)
from tierra_plus.forward.tabulated import bin_tabulated_spectrum
from tierra_plus.models import Spectrum


def test_shifted_integral_matches_direct_piecewise_linear_integration():
    wave = np.array([0.5, 0.8, 1.4, 2.0, 2.8, 3.5])
    depth = np.array([1.0, 2.0, 0.4, 3.0, 1.3, 2.8])
    edges = np.array([1.0, 1.6, 2.3, 3.0])
    shifts = np.array([-0.2, 0, 0.15])
    actual = shifted_bin_integrals(wave, depth, edges, shifts)
    expected = [
        bin_tabulated_spectrum(Spectrum(tuple(wave), tuple(depth)), edges + shift).flux
        for shift in shifts
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize(
    "wave,depth,edges,shifts",
    [
        ([1, 2], [1, 2], [1, 2], [0.1]),
        ([2, 1], [1, 2], [1, 2], [0]),
        ([1, 2], [1, float("nan")], [1, 2], [0]),
        ([1, 2], [1], [1, 2], [0]),
        ([1, 2], [1, 2], [2, 1], [0]),
    ],
)
def test_invalid_binning_rejected(wave, depth, edges, shifts):
    with pytest.raises(ValueError):
        shifted_bin_integrals(wave, depth, edges, shifts)


def test_exact_posterior_matches_direct_offset_integrated_gaussians():
    spectra = np.array([[[0, 1, 0], [0, 1.5, 0]], [[0, 3, 1], [0, 2.5, 1]]], dtype=float)
    labels = np.array([-1.0, 1.0])
    sigma = np.array([0.3, 0.4, 0.5])
    y = np.array([[0.2, 1.1, 0.1], [0.4, 3.3, 1.4]])
    expected = []
    for observation in y:
        likelihood = []
        for model in spectra.reshape(-1, 3):
            residual = observation - model
            offset = np.sum(residual / sigma**2) / np.sum(1 / sigma**2)
            likelihood.append(np.exp(-0.5 * np.sum(((residual - offset) / sigma) ** 2)))
        probability = np.array(likelihood).reshape(2, 2).sum(axis=1)
        probability /= probability.sum()
        expected.append(probability @ labels)
    grid = FiniteSpectralGrid(spectra, labels, sigma)
    result = grid.infer(y, chunk_size=1)
    np.testing.assert_allclose(result["mean"], expected, atol=1e-13)
    shifted = grid.infer(y + np.array([[7], [-3]]))
    for key in result:
        np.testing.assert_allclose(result[key], shifted[key], atol=1e-11)
    for nuisance in range(2):
        conditional = FiniteSpectralGrid(spectra[:, nuisance : nuisance + 1], labels, sigma).infer(
            y
        )
        assert np.all(result["envelope_lower"] <= conditional["lower"])
        assert np.all(result["envelope_upper"] >= conditional["upper"])


def test_degenerate_spectra_preserve_prior_uncertainty():
    grid = FiniteSpectralGrid(np.ones((3, 2, 4)), np.array([-1, 0, 2]), np.ones(4))
    result = grid.infer(np.ones((2, 4)))
    np.testing.assert_allclose(result["mean"], 1 / 3)
    np.testing.assert_array_equal(result["lower"], [-1, -1])
    np.testing.assert_array_equal(result["upper"], [2, 2])


def test_extremely_unlikely_nuisance_still_has_a_valid_conditional_interval():
    spectra = np.array([[[0, 1, 0], [0, 1000, 0]], [[0, 2, 0], [0, 1002, 0]]], dtype=float)
    result = FiniteSpectralGrid(spectra, np.array([0, 1]), np.ones(3)).infer([[0, 2, 0]])
    assert all(np.isfinite(value).all() for value in result.values())
    assert result["envelope_lower"][0] == 0


def test_offset_projection_and_calibration_order_statistic():
    sigma = np.array([0.1, 0.2, 0.4])
    x = np.array([[2, 3, 5], [8, 4, 1]], dtype=float)
    np.testing.assert_allclose(remove_offset(x, sigma), remove_offset(x + 100, sigma), atol=1e-12)
    assert calibration_radius(np.arange(1, 11)) == 10
    assert calibration_radius(-np.arange(1, 11)) == 10
    with pytest.raises(ValueError):
        calibration_radius([1, 2])


@pytest.mark.parametrize("sigma", [[0, 1, 1], [-1, 1, 1], [1, 1], [1, float("nan"), 1]])
def test_grid_rejects_invalid_errors(sigma):
    with pytest.raises(ValueError):
        FiniteSpectralGrid(np.zeros((2, 2, 3)), np.array([0, 1]), np.array(sigma))


def test_metric_rejects_empty_selection_as_a_reliability_pass():
    from examples.physical_grid_reliability import metrics, reliable

    result = {"mean": np.array([0.0]), "lower": np.array([-1.0]), "upper": np.array([1.0])}
    row = metrics(np.array([0.0]), np.array([7]), result, 0.6)
    assert row["coverage90"] == 1
    assert row["acceptance"] == 0
    assert row["accepted_coverage90"] is None
    assert not reliable(row, {"coverage_floor": 0.85, "acceptance_floor": 0.7})


def test_final_decision_applies_frozen_seed_count_and_smoke_never_passes():
    from examples.physical_grid_reliability import decisions

    good = {"gates": {"augmented_network": True, "classical_reliability": {"grid": True}}}
    bad = {"gates": {"augmented_network": False, "classical_reliability": {"grid": False}}}
    assert not decisions([good, good, good, bad, bad], 4, False)["augmented_network_pass"]
    assert decisions([good, good, good, good, bad], 4, False)["classical_pass"]["grid"]
    assert not decisions([good] * 5, 4, True)["augmented_network_pass"]


def test_main_refuses_untracked_source_before_loading_data(monkeypatch, tmp_path):
    from examples import physical_grid_reliability as runner

    def git(command, **kwargs):
        return "?? untracked_executable.py\n" if "status" in command else "sourcehash\n"

    monkeypatch.setattr(runner.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="clean committed"):
        runner.run(tmp_path / "missing", tmp_path / "out")


def test_smoke_never_evaluates_test_or_calibration_states(monkeypatch, tmp_path):
    from examples import physical_grid_reliability as runner

    manifest = {
        "family": "rcte",
        "splits": {"train": list(range(32)), "validation": [32], "test": [33]},
    }
    monkeypatch.setattr(runner, "load_grid", lambda path: (manifest, {}))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "sourcehash\n")
    monkeypatch.setattr(runner.platform, "platform", lambda: "test-platform")
    (tmp_path / "manifest.json").write_text("{}")

    def fake_seed(config, seed, actual, arrays, external, output):
        assert set(actual["splits"]["validation"]).issubset(manifest["splits"]["train"])
        assert set(actual["splits"]["test"]).issubset(manifest["splits"]["train"])
        assert external is None
        return {"gates": {"augmented_network": True, "classical_reliability": {"grid": True}}}

    monkeypatch.setattr(runner, "run_seed", fake_seed)
    result = runner.run(tmp_path, tmp_path / "out", smoke=True)
    assert not result["decisions"]["augmented_network_pass"]
    with pytest.raises(ValueError, match="never open external"):
        runner.run(tmp_path, tmp_path / "other", external_grid=tmp_path, smoke=True)
