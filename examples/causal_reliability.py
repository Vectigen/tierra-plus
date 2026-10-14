"""Frozen causal reliability experiment; run as ``python -m examples.causal_reliability``.

No observed depth is used as a training label. Public assets only anchor the
wavelength grid, noise scale and template shape. All latent labels are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

from tierra_plus.diagnostics.linear import (
    constrained_linear_estimator,
    gaussian_posterior_operator,
    nuisance_projected_gls,
    spectral_design,
)
from tierra_plus.diagnostics.network import TanhRegressor, fit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/configs/causal-reliability-v1.json"


def array_hash(value: np.ndarray) -> str:
    """Hash shape, dtype and exact C-order bytes, not a lossy JSON rendering."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype.str}:{value.shape}:".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config: dict) -> None:
    for key in (
        "train_count",
        "validation_count",
        "discovery_count",
        "test_count",
        "hidden_width",
        "training_steps",
        "batch_size",
        "random_direction_count",
    ):
        if type(config[key]) is not int or config[key] < 2:
            raise ValueError(f"{key} must be an integer >= 2")
    seeds = config["training_seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if type(config["data_seed"]) is not int or config["data_seed"] < 0:
        raise ValueError("Data seed must be a nonnegative integer")
    correlations = config["test_correlations"]
    if (
        len(set(correlations)) != len(correlations)
        or config["training_correlation"] not in correlations
    ):
        raise ValueError("Distinct test correlations must include the training correlation")
    if 0.0 not in correlations or -config["training_correlation"] not in correlations:
        raise ValueError("Independent and reversed test priors are required")
    if any(not np.isfinite(r) or abs(r) >= 1 for r in correlations):
        raise ValueError("Correlations must lie strictly between -1 and 1")
    if not config["overlaps"] or any(not np.isfinite(c) or abs(c) > 1 for c in config["overlaps"]):
        raise ValueError("Spectral overlaps must lie in [-1, 1]")
    if config["interval_probability"] != 0.9:
        raise ValueError("This protocol reports nominal 90% intervals")
    for key in (
        "target_snr",
        "nuisance_snr",
        "learning_rate",
        "ridge_penalty",
        "intervention_half_width",
        "discovery_intervention_half_width",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    widths = config["held_out_intervention_half_widths"]
    if (
        not widths
        or any(not np.isfinite(w) or w <= 0 for w in widths)
        or config["intervention_half_width"] not in widths
    ):
        raise ValueError("Held-out widths must be positive and include the primary width")


def make_split(config: dict, seed: int, split: int, count: int, bins: int) -> dict:
    """Independent PCG64 streams by split; shared across overlap comparisons."""
    words = [config["data_seed"], seed, split]
    rng = np.random.default_rng(np.random.SeedSequence(words))
    result = {
        "a": rng.standard_normal(count),
        "z": rng.standard_normal(count),
        "noise": rng.standard_normal((count, bins)),
        "seed_words": words,
    }
    result["hashes"] = {k: array_hash(result[k]) for k in ("a", "z", "noise")}
    return result


def observations(design: np.ndarray, split: dict, rho: float) -> np.ndarray:
    b = rho * split["a"] + math.sqrt(1 - rho**2) * split["z"]
    return split["a"][:, None] * design[:, 0] + b[:, None] * design[:, 1] + split["noise"]


def intervention_pair(
    design: np.ndarray, split: dict, component: int, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """The other latent and exact noise bytes are held fixed in each pair."""
    if component not in (0, 1) or not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("Choose a valid component and positive intervention width")
    other = split["z"] if component == 0 else split["a"]
    base = other[:, None] * design[:, 1 - component] + split["noise"]
    return base - half_width * design[:, component], base + half_width * design[:, component]


def wilson(successes: int, count: int) -> list[float]:
    if not 0 <= successes <= count or count < 1:
        raise ValueError("Invalid Bernoulli count")
    z = NormalDist().inv_cdf(0.975)
    p = successes / count
    denominator = 1 + z**2 / count
    center = (p + z**2 / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z**2 / (4 * count**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metrics(truth: np.ndarray, prediction: np.ndarray, sd: float) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError("Metrics require matching nonempty vectors")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not 0 < sd < np.inf:
        raise ValueError("Metrics require finite predictions and positive uncertainty")
    error = prediction - truth
    threshold = NormalDist().inv_cdf(0.95) * sd
    covered = int(np.count_nonzero(np.abs(error) <= threshold))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "gaussian_nll": float(
            0.5 * np.mean((error / sd) ** 2) + math.log(sd) + 0.5 * math.log(2 * math.pi)
        ),
        "coverage90": covered / len(truth),
        "coverage_count": covered,
        "coverage95_wilson": wilson(covered, len(truth)),
        "count": len(truth),
        "interval_width90": 2 * threshold,
        "predictive_sd": sd,
    }


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    center, mean = x.mean(axis=0), float(y.mean())
    xc = x - center
    weight = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T @ (y - mean))
    return weight, float(mean - center @ weight)


def hypothesis_gates(rows: list[dict], effects: dict, config: dict) -> dict:
    """Frozen finite decision rules, evaluated separately for each training seed."""
    rho = config["training_correlation"]
    lookup = {(r["method"], r["rho"]): r for r in rows}
    threshold = config["thresholds"]
    iid, reverse = lookup["network", rho], lookup["network", -rho]
    quality = (
        iid["rmse"] <= threshold["iid_rmse_ratio_to_bayes_max"] * lookup["bayes_train", rho]["rmse"]
    )
    h1 = (
        quality
        and reverse["rmse"] >= threshold["reversal_rmse_ratio_min"] * iid["rmse"]
        and reverse["coverage95_wilson"][1] < threshold["reversal_coverage_upper_max"]
    )
    original, removed = effects["network"], effects["hidden_removed"]
    floor = threshold["response_denominator_floor"]
    valid_denominators = original["nuisance_rms"] > floor and original["signal_signed"] > floor
    nuisance_ratio = (
        removed["nuisance_rms"] / original["nuisance_rms"] if valid_denominators else None
    )
    signal_ratio = (
        removed["signal_signed"] / original["signal_signed"] if valid_denominators else None
    )
    frontier_ratio = (
        lookup["hidden_removed", rho]["rmse"] / lookup["constrained_linear", rho]["rmse"]
    )
    h2 = (
        quality
        and valid_denominators
        and nuisance_ratio <= threshold["nuisance_response_ratio_max"]
        and signal_ratio >= threshold["signal_response_ratio_min"]
        and frontier_ratio <= threshold["iid_rmse_ratio_to_constrained_linear_max"]
    )
    # Empirical dominance is descriptive, without selecting a winning random direction.
    comparators = [
        name
        for name in effects
        if name in ("ridge", "decorrelated_network", "gls") or name.startswith("random_")
    ]
    dominates = [
        name
        for name in comparators
        if effects[name]["nuisance_rms"] <= removed["nuisance_rms"]
        and effects[name]["signal_signed"] >= removed["signal_signed"]
        and lookup[name, rho]["rmse"] <= lookup["hidden_removed", rho]["rmse"]
    ]
    h2_status = (
        "inconclusive_fit"
        if not quality
        else (
            "inconclusive_denominator"
            if not valid_denominators
            else "supported" if h2 else "not_supported"
        )
    )
    return {
        "iid_quality_gate": quality,
        "h1_supported": h1,
        "h1_status": "supported" if h1 else "not_supported" if quality else "inconclusive_fit",
        "h2_supported": bool(h2),
        "h2_status": h2_status,
        "h2_denominators_valid": valid_denominators,
        "nuisance_response_ratio": nuisance_ratio,
        "signal_response_ratio": signal_ratio,
        "iid_rmse_ratio_to_constrained_linear": frontier_ratio,
        "simple_baselines_dominating_hidden_removal": dominates,
        "h2_passes_and_not_dominated": bool(h2 and not dominates),
    }


def run_case(design: np.ndarray, config: dict, seed: int, overlap: float, output: Path) -> dict:
    start = time.perf_counter()
    splits = {
        name: make_split(config, seed, i, config[f"{name}_count"], len(design))
        for i, name in enumerate(("train", "validation", "discovery", "test"), start=1)
    }
    train, val, discovery, test = (splits[k] for k in ("train", "validation", "discovery", "test"))
    rho = config["training_correlation"]
    x_train = observations(design, train, rho)
    arguments = {
        "seed": seed,
        "hidden": config["hidden_width"],
        "steps": config["training_steps"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
    }
    network = fit(x_train, train["a"], **arguments)
    independent = fit(observations(design, train, 0.0), train["a"], **arguments)
    # Artifact reload is verified before predictions used for metrics.
    weights = {
        f"{name}_{key}": value
        for name, model in (("network", network), ("decorrelated_network", independent))
        for key, value in model.state_dict().items()
    }
    np.savez_compressed(output / "weights.npz", **weights)
    with np.load(output / "weights.npz", allow_pickle=False) as saved:
        for name, model in (("network", network), ("decorrelated_network", independent)):
            restored = TanhRegressor.from_state_dict(
                {key: saved[f"{name}_{key}"] for key in model.state_dict()}
            )
            if not np.array_equal(model.predict(x_train[:16]), restored.predict(x_train[:16])):
                raise ValueError("Model artifact round trip changed predictions")
    gain, covariance = gaussian_posterior_operator(design, rho=rho)
    independent_gain, _ = gaussian_posterior_operator(design, rho=0.0)
    ridge_weight, ridge_bias = ridge_fit(x_train, train["a"], config["ridge_penalty"])
    response_cap = config["thresholds"]["nuisance_response_ratio_max"] * abs(gain[0] @ design[:, 1])
    constrained, constrained_risk = constrained_linear_estimator(design, rho, response_cap)
    methods = {
        "network": network.predict,
        "decorrelated_network": independent.predict,
        "ridge": lambda x: x @ ridge_weight + ridge_bias,
        "bayes_train": lambda x: x @ gain[0],
        "constrained_linear": lambda x: x @ constrained,
    }
    analytic_sd = {"bayes_train": float(np.sqrt(covariance[0, 0]))}
    omitted = {}
    try:
        _, gls_variance = nuisance_projected_gls(design, np.zeros(len(design)))
        methods["gls"] = lambda x: nuisance_projected_gls(design, x)[0]
        analytic_sd["gls"] = math.sqrt(gls_variance)
    except ValueError as error:
        omitted["gls"] = str(error)
    low, high = intervention_pair(design, discovery, 1, config["discovery_intervention_half_width"])
    direction, center = network.hidden_nuisance_direction(
        low, high, np.full(len(low), 2 * config["discovery_intervention_half_width"])
    )
    directions = {"hidden_removed": direction}
    rng = np.random.default_rng(np.random.SeedSequence([config["data_seed"], seed, 500]))
    for i in range(config["random_direction_count"]):
        vector = rng.standard_normal(config["hidden_width"])
        directions[f"random_hidden_{i:02d}"] = vector / np.linalg.norm(vector)
    for name, vector in directions.items():
        methods[name] = lambda x, v=vector: network.patched_predict(x, v, center)
    validation_correlation = {
        name: 0.0 if name == "decorrelated_network" else rho for name in methods
    }
    validation_errors = {
        name: predict(observations(design, val, validation_correlation[name])) - val["a"]
        for name, predict in methods.items()
    }
    validation_rms = {
        name: float(np.sqrt(np.mean(error**2))) for name, error in validation_errors.items()
    }
    deviations = {name: analytic_sd.get(name, validation_rms[name]) for name in methods}
    raw = {
        "design": design,
        "test_a": test["a"],
        "test_z": test["z"],
        "ridge_weight": ridge_weight,
        "ridge_bias": np.array([ridge_bias]),
        "hidden_center": center,
        "constrained_weight": constrained,
        **{f"direction_{k}": v for k, v in directions.items()},
    }
    rows = []
    for j, test_rho in enumerate(config["test_correlations"]):
        x_test = observations(design, test, test_rho)
        deployment_gain, deployment_cov = gaussian_posterior_operator(design, rho=test_rho)
        predictors = {**methods, "bayes_test": lambda x, g=deployment_gain[0]: x @ g}
        sd = {**deviations, "bayes_test": float(np.sqrt(deployment_cov[0, 0]))}
        raw[f"test_example_spectra_{j}"] = x_test[:8]
        for name, predict in predictors.items():
            prediction = predict(x_test)
            rows.append(
                {"method": name, "rho": test_rho, **metrics(test["a"], prediction, sd[name])}
            )
            if not name.startswith("random_"):
                raw[f"prediction_{j}_{name}"] = prediction
    # Interventions always use the independent deployment prior as bayes_test reference.
    effects = {}
    methods["bayes_test"] = lambda x: x @ independent_gain[0]
    width = config["intervention_half_width"]
    nlow, nhigh = intervention_pair(design, test, 1, width)
    slow, shigh = intervention_pair(design, test, 0, width)
    large_width = max(config["held_out_intervention_half_widths"])
    lown, highn = intervention_pair(design, test, 1, large_width)
    intact = observations(design, test, 0.0)
    removed = intact - test["a"][:, None] * design[:, 0]
    nuisance_responses = {}
    for name, predict in methods.items():
        nuisance = (predict(nhigh) - predict(nlow)) / (2 * width)
        signal = (predict(shigh) - predict(slow)) / (2 * width)
        large = (predict(highn) - predict(lown)) / (2 * large_width)
        removal = predict(intact) - predict(removed)
        nuisance_responses[name] = nuisance
        effects[name] = {
            "nuisance_rms": float(np.sqrt(np.mean(nuisance**2))),
            "nuisance_signed": float(nuisance.mean()),
            "signal_signed": float(signal.mean()),
            "signal_rms": float(np.sqrt(np.mean(signal**2))),
            "nuisance_rms_large": float(np.sqrt(np.mean(large**2))),
            "evidence_removal_rms": float(np.sqrt(np.mean(removal**2))),
        }
        if not name.startswith("random_"):
            raw.update(
                {
                    f"nuisance_{name}": nuisance,
                    f"signal_{name}": signal,
                    f"nuisance_large_{name}": large,
                    f"evidence_removal_{name}": removal,
                }
            )
    discrepancy = float(
        np.sqrt(np.mean((nuisance_responses["network"] - nuisance_responses["bayes_train"]) ** 2))
    )
    gates = hypothesis_gates(rows, effects, config)
    gates["oracle_effect_discrepancy"] = discrepancy
    gates["prior_oracle_explains_effect_within_tolerance"] = (
        gates["iid_quality_gate"]
        and discrepancy <= config["thresholds"]["oracle_effect_discrepancy_max"]
    )
    np.savez_compressed(output / "arrays.npz", **raw)
    result = {
        "seed": seed,
        "overlap": overlap,
        "metrics": rows,
        "interventions": effects,
        "gates": gates,
        "omitted": omitted,
        "constrained_linear_expected_mse": constrained_risk,
        "constrained_linear_response_cap": float(response_cap),
        "validation_residual_rms": validation_rms,
        "validation_bias": {name: float(error.mean()) for name, error in validation_errors.items()},
        "validation_correlation": validation_correlation,
        "predictive_sd": deviations,
        "elapsed_seconds": time.perf_counter() - start,
        "split_manifest": {
            name: {"seed_words": s["seed_words"], "hashes": s["hashes"], "count": len(s["a"])}
            for name, s in splits.items()
        },
        "raw_array_hashes": {name: array_hash(value) for name, value in raw.items()},
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("arrays.npz", "weights.npz")
        },
    }
    write_json(output / "summary.json", result)
    return result


def run(config: dict, output: Path, *, smoke: bool = False) -> dict:
    validate_config(config)
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if dirty and not smoke:
        raise ValueError("Commit the experiment source and protocol before a main run")
    from examples.wasp39_validation import load_inputs

    output.mkdir(parents=True, exist_ok=False)
    observed, template, inputs = load_inputs()
    source_paths = [
        "examples/causal_reliability.py",
        "examples/wasp39_validation.py",
        "src/tierra_plus/diagnostics/linear.py",
        "src/tierra_plus/diagnostics/network.py",
        "src/tierra_plus/forward/tabulated.py",
        "src/tierra_plus/models.py",
        "docs/causal-reliability-protocol.md",
        "examples/configs/causal-reliability-v1.json",
    ]
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    report = {
        "protocol": config["protocol"],
        "smoke": smoke,
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": sources,
        "source_dirty": dirty,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "thread_settings": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "data": inputs,
        "runs": [],
        "complete": False,
        "scope": "Synthetic template-amplitude model organism; not molecular inference.",
        "replay": "PCG64 SeedSequence words + source/version + split hashes regenerate inputs. "
        "Raw float64 predictions, intervention effects and weights are retained; "
        "random-control predictions are reproducible from retained directions/weights.",
    }
    for overlap in config["overlaps"]:
        design = spectral_design(
            template,
            np.asarray(observed.uncertainty),
            overlap,
            config["target_snr"],
            config["nuisance_snr"],
        )
        for seed in config["training_seeds"]:
            directory = output / f"overlap-{overlap:g}-seed-{seed}"
            directory.mkdir()
            result = run_case(design, config, seed, overlap, directory)
            result["artifact_directory"] = directory.name
            report["runs"].append(result)
            write_json(output / "report.json", report)
            print(
                f"overlap={overlap:g} seed={seed}: quality={result['gates']['iid_quality_gate']} "
                f"H1={result['gates']['h1_supported']} H2={result['gates']['h2_supported']}",
                flush=True,
            )
    primary = [r for r in report["runs"] if r["overlap"] == 0.0]
    required = config["thresholds"]["required_training_seeds"]
    report["decisions"] = {}
    for key in (
        "h1_supported",
        "h2_supported",
        "h2_passes_and_not_dominated",
        "prior_oracle_explains_effect_within_tolerance",
    ):
        eligible = sum(
            r["gates"]["iid_quality_gate"]
            and (r["gates"]["h2_denominators_valid"] if key.startswith("h2") else True)
            for r in primary
        )
        passing = sum(bool(r["gates"][key]) for r in primary)
        supported = not smoke and passing >= required
        status = (
            "smoke_only"
            if smoke
            else (
                "supported"
                if supported
                else "inconclusive" if eligible < required else "not_supported"
            )
        )
        report["decisions"][key] = {
            "passing_runs": passing,
            "required_runs": required,
            "eligible_runs": eligible,
            "inconclusive_runs": len(primary) - eligible,
            "supported": supported,
            "status": status,
        }
    report["complete"] = True
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="Cheap plumbing check, not the experiment"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(
            training_seeds=[11],
            train_count=128,
            validation_count=32,
            discovery_count=32,
            test_count=64,
            training_steps=4,
            batch_size=32,
            hidden_width=8,
            random_direction_count=2,
        )
    run(config, args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
