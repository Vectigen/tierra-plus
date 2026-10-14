"""Replay saved predictions and regenerate hashed noise without retraining.

Run ``uv run python -m examples.replay_physical_grid`` from a source checkout.
This verifies the public record; it never changes an experiment artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from tierra_plus.diagnostics.network import TanhRegressor

ROOT = Path(__file__).resolve().parents[1]


def regenerate_draws(report: dict, seed: int, bins: int) -> dict:
    """PCG64 streams are independent by (data seed, fit seed, split number)."""
    config = report["config"]
    rcte = report["inputs"]["rcte_manifest"]
    external = report["inputs"]["photochemical_manifest"]
    samples = {}
    for split, name, metadata, ids, repeat_key in (
        (0, "training", rcte, rcte["splits"]["train"], "training_repeats_per_state"),
        (1, "validation", rcte, rcte["splits"]["validation"], "validation_repeats_per_state"),
        (2, "test", rcte, rcte["splits"]["test"], "test_repeats_per_state"),
        (3, "photochemical", external, external["splits"]["external"], "test_repeats_per_state"),
    ):
        state = np.repeat(ids, config[repeat_key])
        rng = np.random.default_rng(np.random.SeedSequence([config["data_seed"], seed, split]))
        samples[name + "_state"] = state
        samples[name + "_noise"] = rng.standard_normal((len(state), bins))
        if split == 0:
            samples["training_nuisance"] = rng.integers(9, size=len(state))
        if split >= 2:
            labels = np.array([x["log_metallicity"] for x in metadata["states"]])
            samples[name + "_truth"] = labels[state]
    return samples


def replay(results: Path, data: Path) -> dict:
    raw = gzip.decompress((results / "report.json.gz").read_bytes())
    report, record = json.loads(raw), json.loads((results / "record.json").read_text())
    if hashlib.sha256(raw).hexdigest() != record["report_sha256"] or report["smoke"]:
        raise ValueError("Main report integrity check failed")
    config, grids = report["config"], {}
    for family, folder, manifest_key in (
        ("test", "rcte", "rcte_manifest"),
        ("photochemical", "photochemical", "photochemical_manifest"),
    ):
        path = data / folder / "grid.npz"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != report["inputs"][manifest_key]["grid_sha256"]
        ):
            raise ValueError("Prepared input hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            grids[family] = {key: saved[key] for key in saved.files}
    checks, largest = 0, 0.0
    for run in report["runs"]:
        path = results / f"seed-{run['seed']}"
        draws = regenerate_draws(report, run["seed"], len(grids["test"]["sigma"]))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **draws)
        if hashlib.sha256(buffer.getvalue()).hexdigest() != run["artifact_sha256"]["draws.npz"]:
            raise ValueError("Regenerated draw bytes differ; check the recorded replay runtime")
        checks += 1
        for name in ("fit.npz", "predictions.npz"):
            assert (
                hashlib.sha256((path / name).read_bytes()).hexdigest()
                == run["artifact_sha256"][name]
            )
            checks += 1
        with np.load(path / "fit.npz", allow_pickle=False) as saved:
            weights = {key: saved[key] for key in saved.files}
        with np.load(path / "predictions.npz", allow_pickle=False) as saved:
            predictions = {key: saved[key] for key in saved.files}
        for row in run["metrics"]:
            family, condition, method = (row[key] for key in ("family", "condition", "method"))
            prefix = f"{family}__{condition}__{method}__"
            point, lower, upper = (predictions[prefix + key] for key in ("mean", "lower", "upper"))
            truth, state = draws[family + "_truth"], draws[family + "_state"]
            covered = (lower <= truth) & (truth <= upper)
            accepted = upper - lower <= config["width_cap_dex"]
            if prefix + "accept" in predictions:
                accepted &= predictions[prefix + "accept"]
            expected = {
                "rmse_dex": np.sqrt(np.mean((point - truth) ** 2)),
                "bias_dex": np.mean(point - truth),
                "coverage90": covered.mean(),
                "mean_width_dex": np.mean(upper - lower),
                "median_width_dex": np.median(upper - lower),
                "acceptance": accepted.mean(),
                "accepted_count": int(accepted.sum()),
                "accepted_coverage90": covered[accepted].mean() if accepted.any() else None,
                "accepted_rmse_dex": (
                    np.sqrt(np.mean((point[accepted] - truth[accepted]) ** 2))
                    if accepted.any()
                    else None
                ),
                "paired_nuisance_rms_dex": np.sqrt(
                    np.mean((point - predictions[f"{family}__clean__{method}__mean"]) ** 2)
                ),
            }
            for key, value in expected.items():
                assert (
                    row[key] is None
                    if value is None
                    else np.isclose(row[key], value, atol=1e-13, rtol=1e-13)
                )
                checks += 1
            for by_state in row["by_state"]:
                mask = state == by_state["state"]
                assert np.isclose(by_state["coverage90"], covered[mask].mean())
                assert np.isclose(by_state["acceptance"], accepted[mask].mean())
                assert np.isclose(
                    by_state["rmse"], np.sqrt(np.mean((point[mask] - truth[mask]) ** 2))
                )
                selected = mask & accepted
                assert (
                    by_state["accepted_coverage90"] is None
                    if not selected.any()
                    else np.isclose(by_state["accepted_coverage90"], covered[selected].mean())
                )
                checks += 4
            if method in ("network", "augmented_network", "ridge", "augmented_ridge"):
                grid = grids[family]
                intervention = next(
                    x
                    for x in config["primary_conditions"] + [config["secondary_condition"]]
                    if x["name"] == condition
                )
                spectrum = (
                    grid["spectra"][state, intervention["shift_index"]]
                    / (1 + intervention["dilution"])
                    + draws[family + "_noise"] * grid["sigma"]
                )
                u = 1 / grid["sigma"]
                u = u / np.linalg.norm(u)
                x = spectrum / grid["sigma"]
                x = x - (x @ u)[:, None] * u
                kind = "augmented" if method.startswith("augmented") else "nominal"
                if "network" in method:
                    names = ("w1", "b1", "w2", "b2", "x_mean", "x_scale", "y_mean", "y_scale")
                    model = TanhRegressor.from_state_dict(
                        {key: weights[f"network_{kind}_{key}"] for key in names}
                    )
                    replayed = model.predict(x)
                else:
                    replayed = (x - weights[f"ridge_{kind}_mean"]) / weights[
                        f"ridge_{kind}_scale"
                    ] @ weights[f"ridge_{kind}_coefficient"] + weights[f"ridge_{kind}_intercept"][0]
                error = float(np.max(np.abs(point - replayed)))
                largest = max(largest, error)
                assert error < 1e-10
                checks += 1
        primary_names = {x["name"] for x in config["primary_conditions"]}
        primary = [
            x for x in run["metrics"] if x["family"] == "test" and x["condition"] in primary_names
        ]

        def reliable(row: dict) -> bool:
            return bool(
                row["coverage90"] >= config["coverage_floor"]
                and row["acceptance"] >= config["acceptance_floor"]
                and row["accepted_coverage90"] is not None
                and row["accepted_coverage90"] >= config["coverage_floor"]
            )

        for method, passed in run["gates"]["classical_reliability"].items():
            assert passed == all(reliable(x) for x in primary if x["method"] == method)
            checks += 1
        groups = {
            method: [x for x in primary if x["method"] == method]
            for method in ("network", "augmented_network", "augmented_ridge")
        }

        def average(method: str, field: str, grouped: dict = groups) -> float:
            return float(np.mean([x[field] for x in grouped[method]]))

        clean = {
            method: next(x["rmse_dex"] for x in rows if x["condition"] == "clean")
            for method, rows in groups.items()
        }
        components = {
            "coverage_width_acceptance": all(reliable(x) for x in groups["augmented_network"]),
            "beats_augmented_ridge_rmse": average("augmented_network", "rmse_dex")
            <= config["neural_rmse_ratio_to_augmented_ridge"]
            * average("augmented_ridge", "rmse_dex"),
            "nuisance_response_reduced": average("augmented_network", "paired_nuisance_rms_dex")
            <= config["neural_nuisance_response_ratio"]
            * average("network", "paired_nuisance_rms_dex"),
            "clean_quality_retained": clean["augmented_network"]
            <= config["neural_clean_rmse_ratio"] * clean["network"]
            + config["neural_clean_rmse_slack_dex"],
        }
        assert components == run["gates"]["neural_components"]
        assert all(components.values()) == run["gates"]["augmented_network"]
        checks += 2
    required = config["required_seed_passes"]
    assert report["decisions"]["augmented_network_pass"] == (
        sum(x["gates"]["augmented_network"] for x in report["runs"]) >= required
    )
    for method, passed in report["decisions"]["classical_pass"].items():
        assert passed == (
            sum(x["gates"]["classical_reliability"][method] for x in report["runs"]) >= required
        )
        checks += 1
    return {
        "checks": checks,
        "max_prediction_replay_absolute_error": largest,
        "source_commit": report["source_commit"],
        "result": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "examples/results/physical-grid")
    parser.add_argument("--data", type=Path, default=ROOT / "examples/data/physical-grid")
    args = parser.parse_args()
    print(json.dumps(replay(args.results, args.data)))
