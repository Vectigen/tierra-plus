"""Recompute every reported selection metric from hashed public score archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify_bootstraps(
    run: dict, old: dict, saved: dict, scores: dict, conditions: list[str]
) -> int:
    """Independently reconstruct the paired physical-state uncertainty intervals."""
    rows = run["paired_state_bootstrap"]
    keys = [(row["family"], row["condition"], row["method"]) for row in rows]
    expected_keys = [
        (family, condition, method)
        for family in ("test", "photochemical")
        for condition in conditions
        for method in ("grid_marginal", "augmented_network")
    ]
    if keys != expected_keys:
        raise ValueError("Bootstrap comparison panel is incomplete, duplicated or reordered")
    checks = 1
    index = 0
    for family_index, family in enumerate(("test", "photochemical")):
        ids = scores[family + "__state"]
        states, inverse = np.unique(ids, return_inverse=True)
        manifest = old["inputs"]["rcte_manifest" if family == "test" else "photochemical_manifest"]
        truth = np.array([state["log_metallicity"] for state in manifest["states"]])[ids]
        for condition_index, condition in enumerate(conditions):
            prefix = family + "__" + condition + "__"
            rng = np.random.default_rng(
                np.random.SeedSequence([910271, run["seed"], family_index, condition_index])
            )
            tie = rng.random(len(ids))
            masks = []
            for detector in ("predictive_max", "chi2"):
                accepted = np.zeros(len(ids), dtype=bool)
                order = np.lexsort((tie, scores[prefix + detector]))
                accepted[order[: int(np.ceil(0.7 * len(ids)))]] = True
                masks.append(accepted)
            for method in ("grid_marginal", "augmented_network"):
                row = rows[index]
                index += 1
                prediction = [
                    saved[prefix + method + "__" + key] for key in ("mean", "lower", "upper")
                ]
                unsafe = (np.abs(prediction[0] - truth) > 0.25) & (
                    prediction[2] - prediction[1] <= 0.6
                )
                # Frozen protocol has 2,000 state-level paired draws. Each method
                # consumes the next draws from the same condition-specific RNG.
                sampled = rng.integers(len(states), size=(2000, len(states)))
                rates = []
                for selected in masks:
                    numerator = np.bincount(
                        inverse, weights=unsafe & selected, minlength=len(states)
                    )
                    denominator = np.bincount(inverse, weights=selected, minlength=len(states))
                    rates.append((numerator[sampled].sum(axis=1), denominator[sampled].sum(axis=1)))
                valid = (rates[0][1] > 0) & (rates[1][1] > 0)
                delta = (
                    rates[0][0][valid] / rates[0][1][valid]
                    - rates[1][0][valid] / rates[1][1][valid]
                )
                first, second = (unsafe[selected].mean() for selected in masks)
                expected = {
                    "candidate": "predictive_max",
                    "reference": "chi2",
                    "physical_states": len(states),
                    "bootstrap_replicates": 2000,
                    "valid_replicates": int(valid.sum()),
                    "unsafe_rate_difference": float(first - second),
                    "difference_95_percentile": np.quantile(delta, [0.025, 0.975]).tolist(),
                    "relative_reduction": float(1 - first / second) if second else None,
                }
                for key, value in expected.items():
                    actual = row[key]
                    equal = (
                        actual == value
                        if isinstance(value, str) or value is None
                        else np.allclose(actual, value, atol=1e-13, rtol=1e-13)
                    )
                    if not equal:
                        raise ValueError(
                            f"Bootstrap mismatch for {family}/{condition}/{method}: {key}"
                        )
                    checks += 1
    return checks


def replay(results: Path, original: Path) -> dict:
    raw = gzip.decompress((results / "report.json.gz").read_bytes())
    report = json.loads(raw)
    record = json.loads((results / "record.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == record["report_sha256"]
    old_raw = gzip.decompress((original / "report.json.gz").read_bytes())
    assert hashlib.sha256(old_raw).hexdigest() == report["original_report_sha256"]
    old = json.loads(old_raw)
    conditions = [
        x["name"]
        for x in old["config"]["primary_conditions"] + [old["config"]["secondary_condition"]]
    ]
    checks = 2
    for run, old_run in zip(report["runs"], old["runs"], strict=True):
        assert run["seed"] == old_run["seed"]
        score_file = results / f"seed-{run['seed']}.npz"
        assert hashlib.sha256(score_file.read_bytes()).hexdigest() == run["scores_sha256"]
        prediction_file = original / f"seed-{run['seed']}" / "predictions.npz"
        assert (
            hashlib.sha256(prediction_file.read_bytes()).hexdigest()
            == old_run["artifact_sha256"]["predictions.npz"]
        )
        checks += 3
        with np.load(score_file, allow_pickle=False) as data:
            scores = {key: data[key] for key in data.files}
        with np.load(prediction_file, allow_pickle=False) as data:
            saved = {key: data[key] for key in data.files}
        checks += verify_bootstraps(run, old, saved, scores, conditions)
        for key, threshold in run["thresholds"].items():
            assert threshold == np.quantile(scores["validation__" + key], 0.9, method="higher")
            checks += 1
        for row in run["metrics"]:
            family, condition, method, detector = (
                row[key] for key in ("family", "condition", "method", "detector")
            )
            family_index = ("test", "photochemical").index(family)
            condition_index = conditions.index(condition)
            prefix = family + "__" + condition + "__"
            ids = scores[family + "__state"]
            manifest = old["inputs"][
                "rcte_manifest" if family == "test" else "photochemical_manifest"
            ]
            truth = np.array([x["log_metallicity"] for x in manifest["states"]])[ids]
            predictions = saved if method in ("grid_marginal", "augmented_network") else scores
            point, lower, upper = [
                predictions[prefix + method + "__" + key] for key in ("mean", "lower", "upper")
            ]
            width = upper - lower
            if row["selection"] == "always":
                accept = np.ones(len(truth), bool)
            elif row["selection"] == "width_cap_0.6":
                accept = width <= 0.6
            elif row["selection"] == "clean_calibration_90_exploratory":
                accept = (scores[prefix + detector] <= run["thresholds"][detector]) & (width <= 0.6)
            elif row["selection"] == "matched_70_exploratory":
                rng = np.random.default_rng(
                    np.random.SeedSequence([910271, run["seed"], family_index, condition_index])
                )
                tie = rng.random(len(truth))
                values = tie if detector == "random" else scores[prefix + detector]
                selected = np.lexsort((tie, values))[: int(np.ceil(0.7 * len(truth)))]
                accept = np.zeros(len(truth), bool)
                accept[selected] = True
            else:
                raise ValueError("Unknown selection")
            errors = point - truth
            covered = (lower <= truth) & (truth <= upper)
            unsafe = (np.abs(errors) > 0.25) & (width <= 0.6)
            expected = {
                "n": len(truth),
                "accepted_count": accept.sum(),
                "acceptance": accept.mean(),
                "coverage90": covered.mean(),
                "mean_width_dex": width.mean(),
                "accepted_coverage90": covered[accept].mean() if accept.any() else None,
                "accepted_rmse_dex": (
                    np.sqrt(np.mean(errors[accept] ** 2)) if accept.any() else None
                ),
                "unsafe_count": unsafe.sum(),
                "unsafe_accepted_count": (unsafe & accept).sum(),
                "unsafe_accepted_rate": unsafe[accept].mean() if accept.any() else None,
            }
            for key, value in expected.items():
                assert (
                    row[key] is None
                    if value is None
                    else np.isclose(row[key], value, atol=1e-13, rtol=1e-13)
                )
                checks += 1
    return {
        "assertions": checks,
        "status": "all_metrics_and_hashes_match",
        "fresh_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "examples/results/spectral-adequacy")
    parser.add_argument("--original", type=Path, default=ROOT / "examples/results/physical-grid")
    args = parser.parse_args()
    print(json.dumps(replay(args.results, args.original), indent=2))


if __name__ == "__main__":
    main()
