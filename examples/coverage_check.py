"""Repeated-noise coverage check for the synthetic linear spectral example.

This holds the physical parameters fixed and varies Gaussian observation noise.
It is a software check, not prior-predictive simulation-based calibration or a
validation of atmospheric retrieval. Every fit and interval is retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
from synthetic_retrieval import run


def weighted_interval(values: np.ndarray, weights: np.ndarray, level: float) -> tuple[float, float]:
    """Invert the weighted empirical CDF for an equal-tail interval."""
    order = np.argsort(values)
    ordered = values[order]
    cumulative = np.cumsum(weights[order]) / weights.sum()
    tail = (1 - level) / 2
    indices = np.searchsorted(cumulative, [tail, 1 - tail], side="left")
    return tuple(float(v) for v in ordered[np.minimum(indices, len(ordered) - 1)])


def wilson(successes: int, count: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial coverage proportion."""
    z = NormalDist().inv_cdf(0.975)
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count**2))
    return float(center - radius / denominator), float(center + radius / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2 or not 0 < args.level < 1 or args.seed < 0:
        parser.error("Require runs >= 2, 0 < level < 1 and seed >= 0")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    names = []
    for index in range(args.runs):
        result = run(args.seed + index)
        raw_name = f"fit-{index:04d}.json"
        (args.output / raw_name).write_text(json.dumps(result, allow_nan=False) + "\n")
        posterior = result["posterior"]
        samples = np.asarray(posterior["samples"])
        weights = np.asarray(posterior["weights"])
        truth = result["inputs"]["truth"]
        names = posterior["parameter_names"]
        intervals = [
            weighted_interval(samples[:, j], weights, args.level) for j in range(len(names))
        ]
        records.append(
            {
                "seed": args.seed + index,
                "raw_file": raw_name,
                "input_sha256": result["input_sha256"],
                "intervals": intervals,
                "covered": [
                    low <= actual <= high
                    for (low, high), actual in zip(intervals, truth, strict=True)
                ],
                "mean_error_in_analytic_sigma": result["mean_error_in_analytic_sigma"],
            }
        )
    summary = {}
    for j, name in enumerate(names):
        successes = sum(record["covered"][j] for record in records)
        summary[name] = {
            "covered": successes,
            "runs": args.runs,
            "coverage": successes / args.runs,
            "wilson_95_interval": wilson(successes, args.runs),
        }
    payload = {
        "kind": "fixed_truth_repeated_noise_software_validation",
        "nominal_interval_level": args.level,
        "first_seed": args.seed,
        "results": summary,
        "replicates": records,
        "limitations": (
            "Synthetic linear model and independent Gaussian noise; small sample; "
            "not atmospheric validation or prior-predictive SBC."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
