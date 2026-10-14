"""Independent dense-Gaussian checks for withheld-wavelength prediction."""

import numpy as np
import pytest

from tierra_plus.diagnostics.adequacy import SpectralAdequacy
from tierra_plus.diagnostics.physical_grid import FiniteSpectralGrid


def test_predictive_density_matches_direct_correlated_gaussian_mixture():
    rng = np.random.default_rng(91)
    spectra = rng.normal(size=(5, 9))
    labels = np.array([0, 0, 1, 2, 2], dtype=float)
    sigma = np.linspace(0.3, 0.8, 9)
    prior = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
    data = rng.normal(size=(3, 9))
    result = SpectralAdequacy(spectra, labels, sigma, prior).evaluate(data, chunk_size=1)
    expected = []
    for observation in data:
        scores = []
        for block in np.array_split(np.arange(9), 4):
            keep = np.ones(9, dtype=bool)
            keep[block] = False
            precision = np.sum(1 / sigma[keep] ** 2)
            covariance = np.diag(sigma[block] ** 2) + np.ones((len(block), len(block))) / precision
            determinant = np.linalg.slogdet(covariance)[1]
            weights, densities = [], []
            for model, mass in zip(spectra, prior, strict=True):
                offset = np.sum((observation - model)[keep] / sigma[keep] ** 2) / precision
                training = np.sum(((observation - model - offset)[keep] / sigma[keep]) ** 2)
                weights.append(mass * np.exp(-training / 2))
                residual = (observation - model - offset)[block]
                exponent = residual @ np.linalg.solve(covariance, residual)
                densities.append(
                    np.exp(-0.5 * (exponent + determinant + len(block) * np.log(2 * np.pi)))
                )
            density = np.dot(weights, densities) / np.sum(weights)
            normalization = len(block) * np.log(2 * np.pi) + 2 * np.log(sigma[block]).sum()
            scores.append((-2 * np.log(density) - normalization) / len(block))
        expected.append(scores)
    np.testing.assert_allclose(result["predictive_blocks"], expected, atol=1e-12)


def test_summaries_match_existing_grid_and_are_offset_and_chunk_invariant():
    rng = np.random.default_rng(19)
    spectra = rng.normal(size=(4, 2, 8))
    labels = np.array([-1, 0, 1, 3], dtype=float)
    sigma = np.linspace(0.5, 1.2, 8)
    data = rng.normal(size=(7, 8))
    old = FiniteSpectralGrid(spectra, labels, sigma).infer(data)
    model = SpectralAdequacy(spectra.reshape(-1, 8), np.repeat(labels, 2), sigma)
    result = model.evaluate(data)
    for key in ("mean", "lower", "upper", "chi2"):
        np.testing.assert_allclose(result[key], old[key], atol=1e-12)
    shifted = model.evaluate(data + np.arange(7)[:, None] * 17, chunk_size=2)
    for key in result:
        np.testing.assert_allclose(result[key], shifted[key], atol=1e-11)
    light = model.evaluate(data, predictive=False)
    assert "predictive_max" not in light
    for key in light:
        np.testing.assert_allclose(light[key], result[key])


def test_union_prior_is_equal_per_family_and_entropy_is_over_unique_labels():
    spectra = np.ones((4, 8))
    labels = np.array([0, 1, 1, 1], dtype=float)
    result = SpectralAdequacy(
        spectra, labels, np.ones(8), np.array([0.5, 1 / 6, 1 / 6, 1 / 6])
    ).evaluate([[1] * 8])
    np.testing.assert_allclose(result["mean"], 0.5)
    np.testing.assert_allclose(result["entropy"], np.log(2))
    assert result["lower"][0] == 0 and result["upper"][0] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocks": 1},
        {"blocks": 9},
        {"blocks": True},
        {"prior": np.array([0.0, 1.0])},
        {"prior": np.array([1.0])},
        {"prior": np.array([1.0, np.inf])},
        {"sigma": np.zeros(8)},
    ],
)
def test_invalid_grids_rejected(kwargs):
    options = {"spectra": np.zeros((2, 8)), "labels": np.array([0.0, 1.0]), "sigma": np.ones(8)}
    options.update(kwargs)
    with pytest.raises(ValueError):
        SpectralAdequacy(**options)


def test_malformed_observations_and_chunk_limits_rejected():
    model = SpectralAdequacy(np.zeros((2, 8)), np.array([0.0, 1.0]), np.ones(8))
    for data in (np.zeros((0, 8)), np.zeros((2, 7)), [[np.nan] * 8]):
        with pytest.raises(ValueError):
            model.evaluate(data)
    for chunk in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            model.evaluate(np.zeros((2, 8)), chunk_size=chunk)


def test_extreme_shared_misfit_preserves_nonuniform_prior():
    model = SpectralAdequacy(
        np.zeros((2, 8)), np.array([0.0, 1.0]), np.ones(8), np.array([0.25, 0.75])
    )
    result = model.evaluate([[1e9, -1e9] * 4])
    np.testing.assert_allclose(result["mean"], 0.75)
    np.testing.assert_allclose(result["entropy"], -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    np.testing.assert_array_equal(result["lower"], [0])
    np.testing.assert_array_equal(result["upper"], [1])


def test_rank_selection_keeps_exact_count_and_uses_independent_ties():
    from examples.diagnose_spectral_adequacy import rank_accept

    score = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    tie = np.array([0.8, 0.2, 0.4, 0.1, 0.3])
    np.testing.assert_array_equal(rank_accept(score, tie, 0.4), [False, True, True, False, False])
    assert rank_accept(score, tie, 0.7).sum() == 4
    with pytest.raises(ValueError):
        rank_accept(score, tie, 0)


def test_cluster_bootstrap_counts_physical_states_and_preserves_pairing():
    from examples.diagnose_spectral_adequacy import cluster_difference

    ids = np.repeat([0, 1, 2], 10)
    unsafe = ids == 2
    selected = np.tile(np.arange(10) < 7, 3)
    result = cluster_difference(ids, unsafe, selected, selected, np.random.default_rng(2))
    assert result["physical_states"] == 3
    assert result["unsafe_rate_difference"] == 0
    assert result["difference_95_percentile"] == [0, 0]


def test_diagnosis_rejects_untracked_source_before_reading_frozen_inputs(monkeypatch, tmp_path):
    from examples import diagnose_spectral_adequacy as runner

    def fake_git(command, **kwargs):
        return "?? untracked.py\n" if "status" in command else "sourcehash\n"

    monkeypatch.setattr(runner.subprocess, "check_output", fake_git)
    with pytest.raises(ValueError, match="clean committed"):
        runner.run(tmp_path / "missing", tmp_path / "missing", tmp_path / "output")
