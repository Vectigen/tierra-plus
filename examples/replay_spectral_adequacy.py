"""Recompute every reported selection metric from hashed public score archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
