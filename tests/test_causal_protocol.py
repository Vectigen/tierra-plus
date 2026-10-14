"""Check experimental controls and artifact accounting, not research success."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest

from examples import causal_reliability as experiment
from tierra_plus.diagnostics import spectral_design
from tierra_plus.diagnostics.network import TanhRegressor
from tierra_plus.models import Spectrum


@pytest.fixture
def tiny_config():
    config = json.loads(experiment.CONFIG.read_text())
    config.update(
        training_seeds=[11],
        train_count=64,
        validation_count=32,
        discovery_count=32,
        test_count=32,
        hidden_width=8,
        training_steps=4,
        batch_size=16,
        random_direction_count=2,
    )
    return config


def test_split_streams_are_independent_replayable_and_preserved_across_prior_changes(tiny_config):
    streams = [experiment.make_split(tiny_config, 11, i, 32, 5) for i in range(1, 5)]
    for field in ("a", "z", "noise"):
        assert len({split["hashes"][field] for split in streams}) == 4
    replay = experiment.make_split(tiny_config, 11, 4, 32, 5)
    for field in ("a", "z", "noise"):
        np.testing.assert_array_equal(replay[field], streams[3][field])
    other_seed = experiment.make_split(tiny_config, 23, 4, 32, 5)
    assert other_seed["hashes"] != replay["hashes"]

    design = spectral_design([1, 2, 3, 2, 1], [1, 2, 1, 2, 1], overlap=0.3)
    saved = {field: replay[field].copy() for field in ("a", "z", "noise")}
    for rho in tiny_config["test_correlations"]:
        observed = experiment.observations(design, replay, rho)
        b = rho * saved["a"] + math.sqrt(1 - rho**2) * saved["z"]
        recovered_noise = observed - saved["a"][:, None] * design[:, 0] - b[:, None] * design[:, 1]
        np.testing.assert_allclose(recovered_noise, saved["noise"], atol=1e-14)
        for field, original in saved.items():
            np.testing.assert_array_equal(replay[field], original)


@pytest.mark.parametrize("component", [0, 1])
def test_intervention_holds_the_other_latent_and_exact_noise_fixed(tiny_config, component):
    design = spectral_design([0, 1, 0, -1], [1, 2, 3, 4], overlap=0.7)
    split = experiment.make_split(tiny_config, 11, 4, 32, 4)
    low, high = experiment.intervention_pair(design, split, component, half_width=2.0)
    other = split["z"] if component == 0 else split["a"]
    expected_center = other[:, None] * design[:, 1 - component] + split["noise"]
    np.testing.assert_allclose((low + high) / 2, expected_center, atol=1e-14)
    np.testing.assert_allclose(
        high - low, np.broadcast_to(4 * design[:, component], low.shape), atol=1e-14
    )


def test_metrics_count_coverage_and_match_known_error_distribution():
    result = experiment.metrics(np.zeros(4), np.array([-2, -1, 1, 2]), sd=1.0)
    assert result["count"] == 4
    assert result["coverage_count"] == 2
    assert result["coverage90"] == 0.5
    assert result["rmse"] == pytest.approx(math.sqrt(2.5))
    assert result["bias"] == 0
    assert result["gaussian_nll"] == pytest.approx(1.25 + math.log(2 * math.pi) / 2)
    assert result["interval_width90"] == pytest.approx(2 * NormalDist().inv_cdf(0.95))
    np.testing.assert_allclose(
        result["coverage95_wilson"], [0.15003898915214947, 0.8499610108478506]
    )
    assert experiment.wilson(0, 4)[0] == pytest.approx(0)
    assert experiment.wilson(4, 4)[1] == pytest.approx(1)


def gate_inputs(config):
    rho = config["training_correlation"]
    rows = [
        {"method": "network", "rho": rho, "rmse": 0.3},
        {"method": "network", "rho": -rho, "rmse": 3.0, "coverage95_wilson": [0.2, 0.5]},
        {"method": "bayes_train", "rho": rho, "rmse": 0.3},
        {"method": "hidden_removed", "rho": rho, "rmse": 0.4},
        {"method": "constrained_linear", "rho": rho, "rmse": 0.4},
    ]
    effects = {
        "network": {"nuisance_rms": 0.5, "signal_signed": 1.0},
        "hidden_removed": {"nuisance_rms": 0.05, "signal_signed": 0.9},
    }
    return rows, effects


def test_hypothesis_gates_refuse_a_poorly_fitted_model(tiny_config):
    rows, effects = gate_inputs(tiny_config)
    eligible = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert eligible["h1_supported"] and eligible["h2_supported"]
    assert eligible["h2_status"] == "supported"
    rows[0]["rmse"] = 1.0
    rejected = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert rejected["h1_status"] == "inconclusive_fit"
    assert rejected["h2_status"] == "inconclusive_fit"
    assert not rejected["iid_quality_gate"]
    assert not rejected["h1_supported"]
    assert not rejected["h2_supported"]
    assert not rejected["h2_passes_and_not_dominated"]


@pytest.mark.parametrize(
    "field,value", [("nuisance_rms", 0.0), ("signal_signed", 0.0), ("signal_signed", -1.0)]
)
def test_relative_gates_refuse_nonpositive_denominators(tiny_config, field, value):
    rows, effects = gate_inputs(tiny_config)
    effects["network"][field] = value
    result = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert not result["h2_denominators_valid"]
    assert result["h2_status"] == "inconclusive_denominator"
    assert result["nuisance_response_ratio"] is None
    assert result["signal_response_ratio"] is None
    assert not result["h2_supported"]
    assert not result["h2_passes_and_not_dominated"]


def test_valid_but_ineffective_intervention_is_not_supported(tiny_config):
    rows, effects = gate_inputs(tiny_config)
    effects["hidden_removed"]["nuisance_rms"] = 0.4
    result = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert result["iid_quality_gate"] and result["h2_denominators_valid"]
    assert result["h2_status"] == "not_supported"
    assert not result["h2_supported"]


@pytest.mark.parametrize("method", ["ridge", "random_hidden_00", "random_hidden_09"])
def test_ridge_and_every_random_control_can_dominate_hidden_removal(tiny_config, method):
    rows, effects = gate_inputs(tiny_config)
    rows.append({"method": method, "rho": tiny_config["training_correlation"], "rmse": 0.39})
    effects[method] = {"nuisance_rms": 0.04, "signal_signed": 0.95}
    result = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert result["h2_supported"]
    assert result["h2_status"] == "supported"
    assert method in result["simple_baselines_dominating_hidden_removal"]
    assert not result["h2_passes_and_not_dominated"]


@pytest.mark.parametrize("method", ["ridge", "decorrelated_network", "gls", "random_hidden_00"])
def test_low_error_and_nuisance_response_do_not_dominate_if_signal_is_lost(tiny_config, method):
    rows, effects = gate_inputs(tiny_config)
    rows.append({"method": method, "rho": tiny_config["training_correlation"], "rmse": 0.1})
    effects[method] = {"nuisance_rms": 0.0, "signal_signed": 0.1}
    result = experiment.hypothesis_gates(rows, effects, tiny_config)
    assert method not in result["simple_baselines_dominating_hidden_removal"]
    assert result["h2_passes_and_not_dominated"]


@pytest.mark.parametrize("overlap", [0.0, 1.0])
def test_tiny_case_preserves_methods_metrics_and_reloadable_artifacts(
    tmp_path, tiny_config, overlap
):
    design = spectral_design([1, 2, 1, -1, 0], [1, 2, 1, 3, 2], overlap=overlap)
    result = experiment.run_case(design, tiny_config, 11, overlap, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    required_methods = {
        "network",
        "decorrelated_network",
        "ridge",
        "bayes_train",
        "bayes_test",
        "constrained_linear",
        "hidden_removed",
        "random_hidden_00",
        "random_hidden_01",
    }
    if overlap == 0:
        required_methods.add("gls")
        assert "gls" not in result["omitted"]
    else:
        assert "rank deficient" in result["omitted"]["gls"]
    for rho in tiny_config["test_correlations"]:
        assert {row["method"] for row in result["metrics"] if row["rho"] == rho} == required_methods
    assert set(result["interventions"]) == required_methods

    test = experiment.make_split(tiny_config, 11, 4, 32, len(design))
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as raw:
        for name in raw.files:
            assert experiment.array_hash(raw[name]) == result["raw_array_hashes"][name]
        np.testing.assert_array_equal(raw["test_a"], test["a"])
        np.testing.assert_array_equal(raw["test_z"], test["z"])
        with np.load(tmp_path / "weights.npz", allow_pickle=False) as weights:
            models = {
                name: TanhRegressor.from_state_dict(
                    {
                        key[len(name) + 1 :]: weights[key]
                        for key in weights.files
                        if key.startswith(name + "_")
                    }
                )
                for name in ("network", "decorrelated_network")
            }
        validation = experiment.make_split(tiny_config, 11, 2, 32, len(design))
        for name, model in models.items():
            own_rho = 0.0 if name == "decorrelated_network" else tiny_config["training_correlation"]
            validation_b = own_rho * validation["a"] + math.sqrt(1 - own_rho**2) * validation["z"]
            validation_x = (
                validation["a"][:, None] * design[:, 0]
                + validation_b[:, None] * design[:, 1]
                + validation["noise"]
            )
            validation_error = model.predict(validation_x) - validation["a"]
            assert result["validation_correlation"][name] == own_rho
            assert result["validation_bias"][name] == pytest.approx(validation_error.mean())
            assert result["validation_residual_rms"][name] == pytest.approx(
                np.sqrt(np.mean(validation_error**2))
            )
        for index, rho in enumerate(tiny_config["test_correlations"]):
            b = rho * test["a"] + math.sqrt(1 - rho**2) * test["z"]
            x = test["a"][:, None] * design[:, 0] + b[:, None] * design[:, 1] + test["noise"]
            for name, model in models.items():
                np.testing.assert_array_equal(model.predict(x), raw[f"prediction_{index}_{name}"])
            for row in (row for row in result["metrics"] if row["rho"] == rho):
                name = row["method"]
                if name.startswith("random_"):
                    prediction = models["network"].patched_predict(
                        x, raw[f"direction_{name}"], raw["hidden_center"]
                    )
                else:
                    prediction = raw[f"prediction_{index}_{name}"]
                error = prediction - test["a"]
                threshold = NormalDist().inv_cdf(0.95) * row["predictive_sd"]
                covered = np.count_nonzero(abs(error) <= threshold)
                assert row["rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))
                assert row["bias"] == pytest.approx(np.mean(error))
                assert row["coverage_count"] == covered
                assert row["coverage90"] == covered / len(test["a"])
                assert row["count"] == len(test["a"])
    for filename, digest in result["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_smoke_report_cannot_support_hypotheses_even_if_every_case_passes(
    tmp_path, tiny_config, monkeypatch
):
    tiny_config["overlaps"] = [0.0]
    tiny_config["thresholds"]["required_training_seeds"] = 1
    observed = Spectrum([1, 2, 3], [0.02, 0.02, 0.02], [0.001, 0.001, 0.001])
    monkeypatch.setitem(
        sys.modules,
        "examples.wasp39_validation",
        SimpleNamespace(load_inputs=lambda: (observed, np.array([0.01, 0.02, 0.01]), {})),
    )
    all_supported = {
        "iid_quality_gate": True,
        "h1_supported": True,
        "h1_status": "supported",
        "h2_supported": True,
        "h2_status": "supported",
        "h2_denominators_valid": True,
        "h2_passes_and_not_dominated": True,
        "prior_oracle_explains_effect_within_tolerance": True,
    }
    monkeypatch.setattr(
        experiment,
        "run_case",
        lambda design, config, seed, overlap, output: {
            "seed": seed,
            "overlap": overlap,
            "gates": all_supported,
        },
    )
    monkeypatch.setattr(
        experiment,
        "subprocess",
        SimpleNamespace(
            check_output=lambda command, **kwargs: (
                " M experiment.py\n" if command[1] == "status" else "0" * 40
            )
        ),
    )
    report = experiment.run(tiny_config, tmp_path / "smoke", smoke=True)
    assert report["complete"] and report["smoke"] and report["source_dirty"]
    for decision in report["decisions"].values():
        assert decision["passing_runs"] == decision["required_runs"] == 1
        assert decision["eligible_runs"] == 1
        assert decision["inconclusive_runs"] == 0
        assert decision["status"] == "smoke_only"
        assert not decision["supported"]


def test_dirty_main_run_refuses_before_importing_data_loader(tmp_path, tiny_config, monkeypatch):
    def git_status(command, **kwargs):
        assert command == ["git", "status", "--porcelain"]
        return " M experiment.py\n"

    monkeypatch.setattr(experiment, "subprocess", SimpleNamespace(check_output=git_status))
    # Importing a None module entry raises, independently checking the order of
    # the preflight guard without installing or opening an HDF5 dependency.
    monkeypatch.setitem(sys.modules, "examples.wasp39_validation", None)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Commit the experiment source and protocol"):
        experiment.run(tiny_config, output, smoke=False)
    assert not output.exists()


@pytest.mark.parametrize("quality,denominators", [(False, True), (True, False)])
def test_aggregation_distinguishes_inconclusive_cases_from_failed_hypotheses(
    tmp_path, tiny_config, monkeypatch, quality, denominators
):
    tiny_config["overlaps"] = [0.0]
    tiny_config["thresholds"]["required_training_seeds"] = 1
    observed = Spectrum([1, 2, 3], [0.02, 0.02, 0.02], [0.001, 0.001, 0.001])
    monkeypatch.setitem(
        sys.modules,
        "examples.wasp39_validation",
        SimpleNamespace(load_inputs=lambda: (observed, np.array([0.01, 0.02, 0.01]), {})),
    )
    monkeypatch.setattr(
        experiment,
        "subprocess",
        SimpleNamespace(
            check_output=lambda command, **kwargs: "" if command[1] == "status" else "0" * 40
        ),
    )
    gates = {
        "iid_quality_gate": quality,
        "h1_supported": quality,
        "h1_status": "supported" if quality else "inconclusive_fit",
        "h2_supported": False,
        "h2_status": "inconclusive_denominator" if quality else "inconclusive_fit",
        "h2_denominators_valid": denominators,
        "h2_passes_and_not_dominated": False,
        "prior_oracle_explains_effect_within_tolerance": quality,
    }
    monkeypatch.setattr(
        experiment,
        "run_case",
        lambda design, config, seed, overlap, output: {
            "seed": seed,
            "overlap": overlap,
            "gates": gates,
        },
    )
    report = experiment.run(tiny_config, tmp_path / "aggregate", smoke=False)
    for name, decision in report["decisions"].items():
        eligible = int(quality and (denominators if name.startswith("h2") else True))
        assert decision["eligible_runs"] == eligible
        assert decision["inconclusive_runs"] == 1 - eligible
        if eligible:
            assert decision["supported"] and decision["status"] == "supported"
        else:
            assert not decision["supported"]
            assert decision["status"] == "inconclusive"


def test_test_data_cannot_change_weights_directions_or_validation_scales(
    tmp_path, tiny_config, monkeypatch
):
    design = spectral_design([0, 1, -1, 2], [1, 2, 3, 1], overlap=0.0)
    original_path, changed_path = tmp_path / "original", tmp_path / "changed"
    original_path.mkdir()
    changed_path.mkdir()
    original = experiment.run_case(design, tiny_config, 11, 0.0, original_path)
    make_split = experiment.make_split

    def changed_test(config, seed, split, count, bins):
        values = make_split(config, seed, split, count, bins)
        if split == 4:
            values["a"] += 7
            values["noise"] *= 3
            values["hashes"] = {
                name: experiment.array_hash(values[name]) for name in ("a", "z", "noise")
            }
        return values

    monkeypatch.setattr(experiment, "make_split", changed_test)
    changed = experiment.run_case(design, tiny_config, 11, 0.0, changed_path)
    assert original["validation_residual_rms"] == changed["validation_residual_rms"]
    assert original["validation_bias"] == changed["validation_bias"]
    assert original["validation_correlation"] == changed["validation_correlation"]
    assert original["metrics"] != changed["metrics"]
    with (
        np.load(original_path / "weights.npz", allow_pickle=False) as first,
        np.load(changed_path / "weights.npz", allow_pickle=False) as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    with (
        np.load(original_path / "arrays.npz", allow_pickle=False) as first,
        np.load(changed_path / "arrays.npz", allow_pickle=False) as second,
    ):
        frozen = [
            key
            for key in first.files
            if key.startswith("direction_")
            or key in ("hidden_center", "ridge_weight", "ridge_bias", "constrained_weight")
        ]
        for key in frozen:
            np.testing.assert_array_equal(first[key], second[key])
