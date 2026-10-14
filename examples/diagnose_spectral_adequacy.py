"""Exploratory adequacy diagnosis on already exposed physical-grid-v1 data.

No refitting, fresh test set, confirmatory detector result or continuous-label
uncertainty claim. Frozen predictions and exact noise streams are reused.
Run from a clean source checkout with one BLAS thread and --output outside it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from examples.physical_grid_reliability import digest, load_grid, nuisance_grid
from examples.replay_physical_grid import regenerate_draws
from tierra_plus.diagnostics.adequacy import SpectralAdequacy
from tierra_plus.diagnostics.physical_grid import calibration_radius

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("grid_marginal", "augmented_network")
SCORES = ("predictive_max", "chi2", "fitted_block_max", "entropy", "width", "family_disagreement")
MATCHED_ACCEPTANCE = 0.7
ERROR_DEX = 0.25
BOOTSTRAPS = 2000


def rank_accept(score: np.ndarray, tie_break: np.ndarray, fraction: float) -> np.ndarray:
    """Equal-count selection; random tie breakers are independent of error labels."""
    if score.ndim != 1 or score.shape != tie_break.shape or not np.isfinite(score).all():
        raise ValueError("Finite equal-length scores and independent tie breakers required")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    accepted = np.zeros(len(score), dtype=bool)
    accepted[np.lexsort((tie_break, score))[: int(np.ceil(len(score) * fraction))]] = True
    return accepted


def summarize(truth: np.ndarray, prediction: dict, accept: np.ndarray) -> dict:
    error = prediction["mean"] - truth
    width = prediction["upper"] - prediction["lower"]
    covered = (truth >= prediction["lower"]) & (truth <= prediction["upper"])
    unsafe = (np.abs(error) > ERROR_DEX) & (width <= 0.6)
    return {
        "n": len(truth),
        "accepted_count": int(accept.sum()),
        "acceptance": float(accept.mean()),
        "coverage90": float(covered.mean()),
        "mean_width_dex": float(width.mean()),
        "accepted_coverage90": float(covered[accept].mean()) if accept.any() else None,
        "accepted_rmse_dex": float(np.sqrt(np.mean(error[accept] ** 2))) if accept.any() else None,
        "unsafe_count": int(unsafe.sum()),
        "unsafe_accepted_count": int((unsafe & accept).sum()),
        "unsafe_accepted_rate": float(unsafe[accept].mean()) if accept.any() else None,
    }


def cluster_difference(
    ids: np.ndarray,
    unsafe: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Paired physical-state bootstrap of accepted unsafe fractions, not IID draws."""
    states, inverse = np.unique(ids, return_inverse=True)
    counts = np.stack(
        [
            np.bincount(inverse, weights=a, minlength=len(states))
            for a in (unsafe & first, first, unsafe & second, second)
        ]
    )
    sampled = rng.integers(len(states), size=(BOOTSTRAPS, len(states)))
    summed = counts[:, sampled].sum(axis=2)
    valid = (summed[1] > 0) & (summed[3] > 0)
    deltas = summed[0, valid] / summed[1, valid] - summed[2, valid] / summed[3, valid]
    first_rate, second_rate = unsafe[first].mean(), unsafe[second].mean()
    return {
        "physical_states": len(states),
        "bootstrap_replicates": BOOTSTRAPS,
        "valid_replicates": int(valid.sum()),
        "unsafe_rate_difference": float(first_rate - second_rate),
        "difference_95_percentile": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "relative_reduction": float(1 - first_rate / second_rate) if second_rate else None,
    }


def build_models(arrays: dict, external: dict, config: dict) -> tuple:
    sigma = arrays["sigma"]
    equilibrium, photo = nuisance_grid(arrays["spectra"], config), nuisance_grid(
        external["spectra"], config
    )
    eq_spectra, ph_spectra = (x.reshape(-1, len(sigma)) for x in (equilibrium, photo))
    eq_labels, ph_labels = (np.repeat(x["labels"], 9) for x in (arrays, external))
    eq = SpectralAdequacy(eq_spectra, eq_labels, sigma)
    ph = SpectralAdequacy(ph_spectra, ph_labels, sigma)
    prior = np.r_[
        np.full(len(eq_labels), 0.5 / len(eq_labels)), np.full(len(ph_labels), 0.5 / len(ph_labels))
    ]
    union = SpectralAdequacy(
        np.vstack((eq_spectra, ph_spectra)), np.r_[eq_labels, ph_labels], sigma, prior
    )
    return eq, ph, union


def diagnostics(y: np.ndarray, models: tuple) -> tuple[dict, dict]:
    eq, ph, union = models
    result = eq.evaluate(y)
    photo = ph.evaluate(y, predictive=False)
    united = union.evaluate(y, predictive=False)
    scores = {key: result[key] for key in SCORES if key in result}
    scores["width"] = result["upper"] - result["lower"]
    scores["family_disagreement"] = np.abs(result["mean"] - photo["mean"])
    scores["predictive_blocks"] = result["predictive_blocks"]
    envelope = {
        "mean": united["mean"],
        "lower": np.minimum(result["lower"], photo["lower"]),
        "upper": np.maximum(result["upper"], photo["upper"]),
    }
    return scores, {
        "equilibrium": result,
        "union_oracle": united,
        "family_envelope_oracle": envelope,
    }


def run_seed(
    old: dict, seed_run: dict, data: dict, models: tuple, results: Path, output: Path
) -> dict:
    seed, config = seed_run["seed"], old["config"]
    sigma = data["test"]["sigma"]
    draws = regenerate_draws(old, seed, len(sigma))
    original = results / f"seed-{seed}" / "predictions.npz"
    if digest(original) != seed_run["artifact_sha256"]["predictions.npz"]:
        raise ValueError("Frozen predictions checksum mismatch")
    with np.load(original, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    validation_ids = draws["validation_state"]
    val_y = data["test"]["spectra"][validation_ids, 1] + draws["validation_noise"] * sigma
    val_scores, val_results = diagnostics(val_y, models)
    thresholds = {key: float(np.quantile(val_scores[key], 0.9, method="higher")) for key in SCORES}
    radius = calibration_radius(
        data["test"]["labels"][validation_ids] - val_results["equilibrium"]["mean"]
    )
    stored = {f"validation__{key}": value for key, value in val_scores.items()}
    rows, comparisons = [], []
    for family_index, family in enumerate(("test", "photochemical")):
        ids, noise = draws[family + "_state"], draws[family + "_noise"]
        truth = draws[family + "_truth"]
        stored[family + "__state"] = ids
        for condition_index, condition in enumerate(
            config["primary_conditions"] + [config["secondary_condition"]]
        ):
            name = condition["name"]
            y = (
                data[family]["spectra"][ids, condition["shift_index"]] / (1 + condition["dilution"])
                + noise * sigma
            )
            scores, references = diagnostics(y, models)
            prefix = family + "__" + name + "__"
            stored.update({prefix + key: value for key, value in scores.items()})
            rng = np.random.default_rng(
                np.random.SeedSequence([910271, seed, family_index, condition_index])
            )
            tie_break = rng.random(len(truth))
            selections = {
                key: rank_accept(scores[key], tie_break, MATCHED_ACCEPTANCE) for key in SCORES
            }
            selections["random"] = rank_accept(tie_break, tie_break, MATCHED_ACCEPTANCE)
            for method in METHODS:
                prediction = {
                    key: saved[prefix + method + "__" + key] for key in ("mean", "lower", "upper")
                }
                if method == "grid_marginal":
                    for key in prediction:
                        np.testing.assert_allclose(
                            references["equilibrium"][key], prediction[key], atol=1e-9, rtol=0
                        )
                for detector, accept in {"always": np.ones(len(truth), bool), **selections}.items():
                    rows.append(
                        {
                            "family": family,
                            "condition": name,
                            "method": method,
                            "detector": detector,
                            "selection": (
                                "always" if detector == "always" else "matched_70_exploratory"
                            ),
                            **summarize(truth, prediction, accept),
                        }
                    )
                for detector in SCORES:
                    accept = (scores[detector] <= thresholds[detector]) & (
                        prediction["upper"] - prediction["lower"] <= 0.6
                    )
                    rows.append(
                        {
                            "family": family,
                            "condition": name,
                            "method": method,
                            "detector": detector,
                            "selection": "clean_calibration_90_exploratory",
                            **summarize(truth, prediction, accept),
                        }
                    )
                unsafe = (np.abs(prediction["mean"] - truth) > ERROR_DEX) & (
                    prediction["upper"] - prediction["lower"] <= 0.6
                )
                comparisons.append(
                    {
                        "family": family,
                        "condition": name,
                        "method": method,
                        "candidate": "predictive_max",
                        "reference": "chi2",
                        **cluster_difference(
                            ids, unsafe, selections["predictive_max"], selections["chi2"], rng
                        ),
                    }
                )
                if method == "grid_marginal":
                    widened = {
                        "mean": prediction["mean"],
                        "lower": np.minimum(prediction["lower"], prediction["mean"] - radius),
                        "upper": np.maximum(prediction["upper"], prediction["mean"] + radius),
                    }
                    references["uniformly_widened_grid"] = widened
            for method, prediction in references.items():
                if method == "equilibrium":
                    continue
                rows.append(
                    {
                        "family": family,
                        "condition": name,
                        "method": method,
                        "detector": "width",
                        "selection": "width_cap_0.6",
                        **summarize(
                            truth, prediction, prediction["upper"] - prediction["lower"] <= 0.6
                        ),
                    }
                )
                stored.update(
                    {
                        prefix + method + "__" + key: prediction[key]
                        for key in ("mean", "lower", "upper")
                    }
                )
    artifact = output / f"seed-{seed}.npz"
    np.savez_compressed(artifact, **stored)
    return {
        "seed": seed,
        "thresholds": thresholds,
        "uniform_widening_radius_dex": radius,
        "metrics": rows,
        "paired_state_bootstrap": comparisons,
        "scores_sha256": digest(artifact),
    }


def run(results: Path, data_path: Path, output: Path) -> dict:
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Diagnosis requires clean committed source")
    if any(
        os.environ.get(key) != "1"
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    ):
        raise ValueError("Use one BLAS thread")
    raw = gzip.decompress((results / "report.json.gz").read_bytes())
    record = json.loads((results / "record.json").read_text())
    if hashlib.sha256(raw).hexdigest() != record["report_sha256"]:
        raise ValueError("Original report hash mismatch")
    old = json.loads(raw)
    if old["smoke"] or old["config"]["protocol"] != "physical-grid-v1":
        raise ValueError("Expected frozen physical-grid-v1 main results")
    data = {}
    for family, folder, key in (
        ("test", "rcte", "rcte_manifest"),
        ("photochemical", "photochemical", "photochemical_manifest"),
    ):
        manifest, arrays = load_grid(data_path / folder)
        if manifest != old["inputs"][key]:
            raise ValueError("Input manifest differs from exposed experiment")
        data[family] = arrays
    if not np.array_equal(data["test"]["sigma"], data["photochemical"]["sigma"]):
        raise ValueError("Chemical families need identical wavelength/error geometry")
    models = build_models(data["test"], data["photochemical"], old["config"])
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "status": "started",
        "source_commit": source,
        "started_utc": datetime.now(UTC).isoformat(),
        "kind": "exploratory_diagnosis_of_exposed_v1_data",
        "fresh_validation": False,
        "original_report_sha256": record["report_sha256"],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "protocol_sha256": digest(ROOT / "docs/spectral-adequacy-diagnosis.md"),
        "method": "four fixed equal-count wavelength blocks, integrated offset, frozen estimators",
        "error_threshold_dex": ERROR_DEX,
        "matched_acceptance": MATCHED_ACCEPTANCE,
        "bootstrap_replicates": BOOTSTRAPS,
        "control_roles": {
            "equilibrium": "full support includes original test states",
            "union_oracle": "equal chemical-family prior; includes all exposed target states",
            "family_envelope_oracle": (
                "union of family-conditional intervals; includes all exposed target states"
            ),
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started, runs = time.monotonic(), []
    try:
        for seed_run in old["runs"]:
            runs.append(run_seed(old, seed_run, data, models, results, output))
            print(
                json.dumps(
                    {
                        "seed_complete": seed_run["seed"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "completed_seeds": [x["seed"] for x in runs],
                },
                indent=2,
            )
            + "\n"
        )
        raise
    report = {
        **provenance,
        "status": "complete",
        "elapsed_seconds": time.monotonic() - started,
        "runs": runs,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "examples/results/physical-grid")
    parser.add_argument("--data", type=Path, default=ROOT / "examples/data/physical-grid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.data, args.output)


if __name__ == "__main__":
    main()
