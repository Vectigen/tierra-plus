"""Render the frozen physical-grid report; this never reruns an estimator."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

METHODS = {
    "network": ("Ordinary neural", "#a75d4a", "-"),
    "augmented_network": ("Augmented neural", "#284e80", "-"),
    "augmented_ridge": ("Augmented ridge", "#5b7c42", "-"),
    "training_grid_marginal": ("Training-support grid", "#87639a", "--"),
    "grid_marginal": ("Full-support grid", "#222222", "--"),
    "grid_envelope": ("Full-support envelope", "#877b64", ":"),
}


def plot(report: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(
        gzip.decompress(report.read_bytes()) if report.suffix == ".gz" else report.read_bytes()
    )
    if data["smoke"]:
        raise ValueError("A smoke run is not scientific evidence")
    conditions = [c["name"] for c in data["config"]["primary_conditions"]]
    labels = ["Clean", "Gray", "Shift −", "Shift +", "Both −", "Both +"]
    families = ["test", "photochemical"] if data["inputs"]["photochemical_manifest"] else ["test"]
    fig, axes = plt.subplots(
        len(families), 2, figsize=(12, 7 if len(families) == 2 else 4.4), squeeze=False
    )
    for row, family in enumerate(families):
        for col, (metric, ylabel) in enumerate(
            [("rmse_dex", "Metallicity RMSE (dex)"), ("coverage90", "90% interval coverage")]
        ):
            ax = axes[row, col]
            for method, (label, color, style) in METHODS.items():
                if metric == "rmse_dex" and method == "grid_envelope":
                    continue
                values = np.array(
                    [
                        [
                            next(
                                r[metric]
                                for r in run["metrics"]
                                if r["family"] == family
                                and r["method"] == method
                                and r["condition"] == condition
                            )
                            for condition in conditions
                        ]
                        for run in data["runs"]
                    ]
                )
                x = np.arange(len(conditions))
                ax.plot(
                    x,
                    values.mean(axis=0),
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    label=label,
                )
                ax.fill_between(
                    x, values.min(axis=0), values.max(axis=0), color=color, alpha=0.09, linewidth=0
                )
            ax.set_xticks(np.arange(len(conditions)), labels, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            if metric == "coverage90":
                ax.axhline(0.9, color="#888888", linewidth=0.8, zorder=0)
                ax.set_ylim(-0.02, 1.03)
            else:
                ax.set_ylim(bottom=0)
            title = (
                "Held-out equilibrium atmospheres"
                if family == "test"
                else "External photochemical atmospheres"
            )
            ax.set_title(title, loc="left", fontsize=11, pad=9)
    handles, names = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        ncol=3,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        "Physical spectral reliability under controlled nuisance shifts",
        fontsize=14,
        x=0.07,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.07,
        0.018,
        "Synthetic noisy observations of published PICASO atmospheres. "
        "Lines: five-seed means; bands: min–max, not confidence intervals.\n"
        "Full-support reference includes the equilibrium test states. "
        "Width, abstention and per-state results are reported separately.",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.07, right=0.98, hspace=0.43, wspace=0.23)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)
