"""Plot a completed causal-reliability report without rerunning any experiment.

Matplotlib is an optional plotting dependency, for example::

    uv run --with matplotlib python examples/plot_causal_reliability.py \
        --report examples/results/causal-reliability/report.json \
        --output examples/results/causal-reliability/summary.png

Both PNG and SVG are written using the output path's stem. Error bars describe the
range across training seeds, not confidence intervals or independent replications.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

COLORS = {
    "network": "#245c88",
    "hidden_removed": "#bf632f",
    "decorrelated_network": "#267c6a",
    "bayes_train": "#805a91",
    "bayes_test": "#363a3e",
    "gls": "#a07c24",
    "constrained_linear": "#af405c",
}
LABELS = {
    "network": "Network",
    "hidden_removed": "Hidden direction removed",
    "decorrelated_network": "Network, independent training prior",
    "bayes_train": "Bayes, training prior",
    "bayes_test": "Bayes, actual test prior (oracle)",
    "gls": "Nuisance-projected GLS",
    "constrained_linear": "Constrained linear predictor",
}
MAIN_METHODS = (
    "network",
    "hidden_removed",
    "decorrelated_network",
    "bayes_train",
    "bayes_test",
)


def _finite(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("report must contain a nonempty runs list")
    identities: set[tuple[int, float]] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be an object")
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("each run must identify its integer training seed")
        overlap = _finite(run.get("overlap"), "overlap")
        identity = seed, overlap
        if identity in identities:
            raise ValueError(f"duplicate run for seed {seed}, overlap {overlap}")
        identities.add(identity)
        if not isinstance(run.get("metrics"), list) or not isinstance(
            run.get("interventions"), dict
        ):
            raise ValueError("each run must contain metrics and interventions")
    return runs


def _metric(run: dict[str, Any], method: str, rho: float, name: str) -> float:
    matches = [
        row
        for row in run["metrics"]
        if row.get("method") == method
        and np.isclose(_finite(row.get("rho"), "rho"), rho, rtol=0.0, atol=1e-10)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} result for seed={run['seed']}, "
            f"overlap={run['overlap']}, rho={rho}; found {len(matches)}"
        )
    return _finite(matches[0].get(name), f"{method}.{name}")


def _intervention(run: dict[str, Any], method: str) -> float:
    return _finite(
        run["interventions"].get(method, {}).get("nuisance_rms"),
        f"{method}.nuisance_rms",
    )


def _mean_range(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    return means, np.vstack([means - values.min(axis=0), values.max(axis=0) - means])


def plot(report: Path, output: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = _load(report)
    zero_runs = sorted(
        [run for run in runs if np.isclose(run["overlap"], 0.0, rtol=0.0, atol=1e-10)],
        key=lambda run: run["seed"],
    )
    if not zero_runs:
        raise ValueError("the report has no overlap-zero runs")
    rhos = sorted(
        {
            _finite(row.get("rho"), "rho")
            for row in zero_runs[0]["metrics"]
            if row.get("method") == "network"
        },
        reverse=True,
    )
    if len(rhos) < 2 or min(rhos) >= 0 or max(rhos) <= 0:
        raise ValueError("the report must include positive and reversed prior correlations")
    iid_rho, reversed_rho = max(rhos), min(rhos)
    overlaps = sorted({_finite(run["overlap"], "overlap") for run in runs})
    if len(overlaps) < 2:
        raise ValueError("the identifiability panel requires multiple overlap conditions")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b6b8ba",
            "axes.labelcolor": "#30363b",
            "xtick.color": "#4a5055",
            "ytick.color": "#4a5055",
            "grid.color": "#e4e6e7",
            "grid.linewidth": 0.65,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.8))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.79, bottom=0.17, hspace=0.52, wspace=0.25)
    fig.suptitle(
        "What changes when the nuisance prior changes?", x=0.085, y=0.98, ha="left", fontsize=19
    )
    fig.text(
        0.085,
        0.945,
        "Controlled template-amplitude experiment · synthetic data on a public spectral grid",
        fontsize=11,
        color="#535b61",
    )
    positions = np.arange(len(rhos), dtype=float)
    labels = [f"ρ = {rho:+.2g}" for rho in rhos]
    for axis, metric_name, title, ylabel in (
        (axes[0, 0], "rmse", "A  Prior shift at zero overlap", "RMSE (target standard deviations)"),
        (
            axes[0, 1],
            "coverage90",
            "B  Nominal intervals under the same shift",
            "90% interval coverage",
        ),
    ):
        for method in MAIN_METHODS:
            values = np.array(
                [[_metric(run, method, rho, metric_name) for rho in rhos] for run in zero_runs]
            )
            means, ranges = _mean_range(values)
            axis.errorbar(
                positions,
                means,
                yerr=ranges,
                color=COLORS[method],
                marker="o",
                markersize=4,
                linewidth=1.6,
                elinewidth=0.8,
                capsize=2.5,
                zorder=3,
            )
            for seed_values in values:
                axis.scatter(
                    positions, seed_values, s=12, color=COLORS[method], alpha=0.28, zorder=2
                )
        axis.set_title(title, loc="left", pad=12)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Test correlation ρ between target and nuisance")
        axis.set_xlim(-0.18, len(rhos) - 0.82)
        axis.grid(axis="y", zorder=0)
        if metric_name == "coverage90":
            axis.axhline(0.9, color="#83878a", linewidth=0.9, linestyle="--", zorder=1)
            axis.text(
                0.985,
                0.91,
                "nominal 90%",
                transform=axis.get_yaxis_transform(),
                ha="right",
                fontsize=8,
            )
            axis.set_ylim(0, 1.025)
        else:
            axis.set_ylim(bottom=0)

    ax = axes[1, 0]
    tradeoff_methods = (
        "network",
        "hidden_removed",
        "decorrelated_network",
        "gls",
        "constrained_linear",
    )
    for run in zero_runs:
        controls = sorted(
            method for method in run["interventions"] if method.startswith("random_hidden_")
        )
        for method in controls:
            ax.scatter(
                _intervention(run, method),
                _metric(run, method, reversed_rho, "rmse"),
                color="#92989d",
                s=15,
                alpha=0.23,
                linewidths=0,
                zorder=1,
            )
    for method in tradeoff_methods:
        x = np.array([_intervention(run, method) for run in zero_runs])
        y = np.array([_metric(run, method, reversed_rho, "rmse") for run in zero_runs])
        ax.scatter(x, y, s=23, color=COLORS[method], alpha=0.5, linewidths=0, zorder=2)
        ax.scatter(
            x.mean(),
            y.mean(),
            s=78,
            color=COLORS[method],
            marker="D",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    ax.set_title("C  Error versus nuisance sensitivity", loc="left", pad=12)
    ax.set_xlabel("Paired nuisance-response RMS (target standard deviations)")
    ax.set_ylabel("Reversed-prior RMSE (target standard deviations)")
    ax.set_xlim(left=-0.03)
    ax.set_ylim(bottom=0)
    ax.grid(zorder=0)
    ax.text(
        0.02,
        0.98,
        "Overlap = 0; same-noise intervention pairs\n"
        "Diamonds: seed means · gray: random hidden removals",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#5f666c",
    )

    ax = axes[1, 1]
    overlap_positions = np.arange(len(overlaps), dtype=float)
    for method in ("network", "bayes_train", "bayes_test"):
        for rho, style, marker in ((iid_rho, "--", "o"), (reversed_rho, "-", "s")):
            means, lows, highs = [], [], []
            for overlap in overlaps:
                condition_runs = [run for run in runs if run["overlap"] == overlap]
                values = np.array([_metric(run, method, rho, "rmse") for run in condition_runs])
                means.append(float(values.mean()))
                lows.append(float(values.mean() - values.min()))
                highs.append(float(values.max() - values.mean()))
            ax.errorbar(
                overlap_positions,
                means,
                yerr=[lows, highs],
                color=COLORS[method],
                linestyle=style,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                elinewidth=0.7,
                capsize=2,
                alpha=0.9,
                zorder=3,
            )
    ax.set_title("D  When signal and nuisance become indistinguishable", loc="left", pad=12)
    ax.set_xlabel("Signal–nuisance shape overlap (conditions equally spaced)")
    ax.set_ylabel("RMSE (target standard deviations)")
    ax.set_xticks(overlap_positions, [f"{overlap:g}" for overlap in overlaps])
    ax.set_xlim(-0.15, len(overlaps) - 0.85)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", zorder=0)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#5d6368",
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"IID prior: ρ = {iid_rho:+.2g}",
            ),
            Line2D(
                [0],
                [0],
                color="#5d6368",
                linestyle="-",
                marker="s",
                markersize=4,
                label=f"Reversed prior: ρ = {reversed_rho:+.2g}",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    legend_methods = (*MAIN_METHODS, "gls", "constrained_linear")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=LABELS[method],
        )
        for method in legend_methods
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.91),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=2.1,
        handlelength=2.2,
    )
    seed_counts = sorted({sum(run["overlap"] == overlap for run in runs) for overlap in overlaps})
    count_text = "/".join(str(count) for count in seed_counts)
    fig.text(
        0.085,
        0.055,
        f"Points/lines show means across {count_text} training seeds; "
        "bars show seed ranges, not confidence intervals.\n"
        "The actual-prior Bayes estimator has privileged prior knowledge. "
        "Shape overlap 1 is non-identifiable without a prior.\n"
        "Template amplitude is not molecular abundance. "
        "This figure does not establish performance on real atmospheric retrievals.",
        fontsize=8.5,
        color="#5f666c",
        linespacing=1.55,
    )
    png, svg = output.with_suffix(".png"), output.with_suffix(".svg")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    for path in plot(arguments.report, arguments.output):
        print(path)


if __name__ == "__main__":
    main()
