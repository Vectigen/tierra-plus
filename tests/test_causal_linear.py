"""Independent mathematical checks of the spectral experiment's controls."""

import numpy as np
import pytest

from tierra_plus.diagnostics import (
    constrained_linear_estimator,
    gaussian_posterior,
    gaussian_posterior_operator,
    nuisance_projected_gls,
    spectral_design,
)


@pytest.mark.parametrize("overlap", [-1.0, -0.8, 0.0, 0.7, 1.0])
def test_spectral_design_preserves_declared_geometry(overlap):
    template = np.array([0.02, 0.021, 0.019, 0.022, 0.02])
    sigma = np.array([1, 2, 1, 3, 2]) * 0.0001
    design = spectral_design(template, sigma, overlap, target_snr=3, nuisance_snr=7)
    np.testing.assert_allclose(np.linalg.norm(design, axis=0), [3, 7], atol=1e-14)
    assert np.dot(design[:, 0], design[:, 1]) / 21 == pytest.approx(overlap, abs=1e-14)
    assert np.dot(design[:, 0], 1 / sigma) == pytest.approx(0, abs=1e-10)
    assert np.linalg.matrix_rank(design) == (1 if abs(overlap) == 1 else 2)
    # A physical continuum-depth offset cannot change the feature direction.
    np.testing.assert_allclose(spectral_design(template + 0.1, sigma, overlap, 3, 7), design)


@pytest.mark.parametrize("rho", [-0.97, 0.0, 0.95])
def test_gaussian_posterior_matches_dense_joint_conditioning(rho):
    rng = np.random.default_rng(12)
    design = rng.normal(size=(9, 2))
    observations = rng.normal(size=(5, 9))
    prior = np.array([[1, rho], [rho, 1]])
    cross = prior @ design.T
    marginal = np.eye(9) + design @ cross
    expected_gain = np.linalg.solve(marginal, cross.T).T
    expected_covariance = prior - expected_gain @ cross.T
    mean, covariance = gaussian_posterior(design, observations, rho=rho)
    np.testing.assert_allclose(mean, observations @ expected_gain.T, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-12, atol=1e-13)
    single_mean, single_covariance = gaussian_posterior(design, observations[0], rho=rho)
    np.testing.assert_allclose(single_mean, mean[0])
    np.testing.assert_allclose(single_covariance, covariance)


def test_gaussian_zero_information_returns_the_prior():
    gain, covariance = gaussian_posterior_operator(np.zeros((3, 2)), rho=0.8)
    np.testing.assert_array_equal(gain, np.zeros((2, 3)))
    np.testing.assert_allclose(covariance, [[1, 0.8], [0.8, 1]])


def test_training_and_deployment_priors_are_distinct_explicit_controls():
    design = spectral_design([0, 1, 0, -1], [1, 1, 1, 1], overlap=0.0)
    nuisance_only = design @ np.array([0.0, 1.0])
    deployment, _ = gaussian_posterior(design, nuisance_only, rho=0)
    training, _ = gaussian_posterior(design, nuisance_only, rho=0.95)
    assert abs(deployment[0]) < 1e-14
    assert training[0] > 0.1


def test_gls_is_invariant_to_paired_nuisance_interventions():
    design = spectral_design([0, 1, 0, -1], [1, 2, 3, 4], overlap=0.8)
    rng = np.random.default_rng(21)
    latent = rng.normal(size=(7, 2))
    noise = rng.normal(size=(7, 4))
    observed = latent @ design.T + noise
    shifted = observed + np.arange(1, 8)[:, None] * design[:, 1]
    estimate, variance = nuisance_projected_gls(design, observed)
    counterfactual, counterfactual_variance = nuisance_projected_gls(design, shifted)
    np.testing.assert_allclose(estimate, counterfactual, atol=1e-13)
    covariance = np.linalg.inv(design.T @ design)
    expected = np.linalg.solve(design.T @ design, design.T @ observed.T)[0]
    np.testing.assert_allclose(estimate, expected, atol=1e-13)
    assert variance == pytest.approx(covariance[0, 0])
    assert counterfactual_variance == variance
    single, single_variance = nuisance_projected_gls(design, observed[0])
    assert single == pytest.approx(estimate[0])
    assert single_variance == variance


def test_overlap_one_is_not_identifiable_even_with_finite_bayesian_posterior():
    design = spectral_design([0, 1, -1], [1, 2, 3], overlap=1)
    with pytest.raises(ValueError, match="not identifiable"):
        nuisance_projected_gls(design, np.zeros(3))
    mean, covariance = gaussian_posterior(design, np.zeros(3), rho=0)
    np.testing.assert_array_equal(mean, [0, 0])
    # The unobserved contrast retains its unit prior variance.
    null_direction = np.array([3, -1]) / np.sqrt(10)
    assert null_direction @ covariance @ null_direction == pytest.approx(1.0)
    assert np.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize("design", [np.zeros((3, 2)), [[1, 0], [2, 0]], [[1, 2]]])
def test_gls_rejects_other_rank_deficient_designs(design):
    with pytest.raises(ValueError, match="not identifiable"):
        nuisance_projected_gls(design, np.zeros(len(design)))


@pytest.mark.parametrize(
    "template,sigma,overlap",
    [
        ([1, 1], [1, 2], 0),
        ([1], [1], 0),
        ([1, 2], [1], 0),
        ([1, 2], [1, 0], 0),
        ([1, 2], [1, -1], 0),
        ([1, np.nan], [1, 1], 0),
        ([1, 2], [1, np.inf], 0),
        ([1, 2], [1, 1], np.nan),
        ([1, 2], [1, 1], 1.01),
        ([[1, 2]], [[1, 1]], 0),
    ],
)
def test_invalid_spectral_design_is_rejected(template, sigma, overlap):
    with pytest.raises(ValueError):
        spectral_design(template, sigma, overlap)


@pytest.mark.parametrize("snr", [0, -1, np.nan, np.inf])
def test_invalid_signal_norm_is_rejected(snr):
    with pytest.raises(ValueError):
        spectral_design([1, 2], [1, 1], 0, target_snr=snr)
    with pytest.raises(ValueError):
        spectral_design([1, 2], [1, 1], 0, nuisance_snr=snr)


@pytest.mark.parametrize("rho", [-1, 1, np.nan, np.inf])
def test_invalid_prior_is_rejected(rho):
    with pytest.raises(ValueError, match="rho"):
        gaussian_posterior(np.eye(2), np.zeros(2), rho=rho)


@pytest.mark.parametrize("design", [[], [1, 2], [[1, 2, 3]], [[1, np.inf]]])
def test_malformed_design_is_rejected(design):
    with pytest.raises(ValueError):
        gaussian_posterior_operator(design)
    with pytest.raises(ValueError):
        nuisance_projected_gls(design, [0])


@pytest.mark.parametrize("observations", [[], [1], [[1]], [[[]]], [0, np.nan], np.zeros((0, 2))])
def test_malformed_observations_are_rejected(observations):
    with pytest.raises(ValueError):
        gaussian_posterior(np.eye(2), observations)
    with pytest.raises(ValueError):
        nuisance_projected_gls(np.eye(2), observations)


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.95])
@pytest.mark.parametrize("cap_fraction", [0.0, 0.2, 0.8])
def test_constrained_linear_frontier_matches_dense_kkt_optimum(rho, cap_fraction):
    design = spectral_design([0, 1, 2, -1, 0], [1, 2, 1, 3, 2], overlap=0.7)
    prior = np.array([[1, rho], [rho, 1]])
    sigma = np.eye(5) + design @ prior @ design.T
    cross = design @ prior[:, 0]
    unconstrained = np.linalg.solve(sigma, cross)
    response = unconstrained @ design[:, 1]
    cap = cap_fraction * abs(response)
    readout, mse = constrained_linear_estimator(design, rho, cap)
    target = np.clip(response, -cap, cap)
    kkt = np.block([[sigma, design[:, 1:2]], [design[:, 1:2].T, np.zeros((1, 1))]])
    expected = np.linalg.solve(kkt, np.append(cross, target))[:-1]
    np.testing.assert_allclose(readout, expected, atol=1e-12, rtol=1e-12)
    assert abs(readout @ design[:, 1]) <= cap + 1e-13
    assert mse == pytest.approx(1 - 2 * readout @ cross + readout @ sigma @ readout)
    # Every feasible perturbation away from this readout increases the risk.
    rng = np.random.default_rng(14)
    nuisance = design[:, 1]
    for perturbation in rng.normal(size=(5, 5)):
        perturbation -= nuisance * (perturbation @ nuisance) / (nuisance @ nuisance)
        alternative = readout + 0.1 * perturbation
        other_mse = 1 - 2 * alternative @ cross + alternative @ sigma @ alternative
        assert other_mse >= mse


def test_inactive_frontier_constraint_returns_bayes_readout_and_posterior_variance():
    design = spectral_design([0, 1, -1], [1, 1, 1], overlap=0.5)
    gain, covariance = gaussian_posterior_operator(design, rho=0.95)
    readout, mse = constrained_linear_estimator(design, 0.95, 10.0)
    np.testing.assert_array_equal(readout, gain[0])
    assert mse == pytest.approx(covariance[0, 0])


def test_rank_one_zero_response_frontier_preserves_unidentifiability():
    design = spectral_design([0, 1, -1], [1, 1, 1], overlap=1)
    readout, mse = constrained_linear_estimator(design, 0.95, 0)
    np.testing.assert_allclose(readout, np.zeros(3), atol=1e-14)
    assert mse == pytest.approx(1.0)


@pytest.mark.parametrize("cap", [-1, np.nan, np.inf])
def test_invalid_frontier_constraint_is_rejected(cap):
    with pytest.raises(ValueError, match="max_nuisance_response"):
        constrained_linear_estimator(np.eye(2), 0, cap)


def test_frontier_rejects_a_zero_nuisance_column():
    with pytest.raises(ValueError, match="nuisance column"):
        constrained_linear_estimator([[1, 0], [2, 0]], 0.5, 0)
