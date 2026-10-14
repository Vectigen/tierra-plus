"""Build and run an explicitly exploratory physical-grid reliability audit.

This demo reuses exposed PICASO RCTE and photochemical assets.  Its disjoint state
subsets exercise the software contract; they are not fresh scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tierra_plus.provenance import hash_path
from tierra_plus.reliability import run_reliability_audit, write_audit_report

ROOT = Path(__file__).resolve().parents[1]
FloatArray = NDArray[np.float64]


def _fingerprint(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _write_npz(path: Path, arrays: dict[str, FloatArray]) -> None:
    """Write stable NPZ bytes for exact replay in the same numerical environment."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            handle = BytesIO()
            np.save(handle, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, handle.getvalue())


def _write_bundle(
    root: Path,
    role: str,
    dataset_id: str,
    source: str,
    state_rows: list[dict[str, Any]],
    arrays: dict[str, FloatArray],
) -> Path:
    bundle = root / dataset_id
    bundle.mkdir()
    _write_npz(bundle / "data.npz", arrays)
    state_ids = [str(row["id"]) for row in state_rows]
    state_fingerprints = [str(row["sha256"]) for row in state_rows]
    manifest = {
        "schema_version": 1,
        "role": role,
        "dataset_id": dataset_id,
        "source": source,
        "source_fingerprint": _fingerprint([source, *state_fingerprints]),
        "state_ids": state_ids,
        "state_fingerprints": state_fingerprints,
        "data_file": "data.npz",
        "data_sha256": hash_path(bundle / "data.npz"),
        "wavelength_unit": "micron",
        "spectral_quantity": "transit_depth",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    physical_root = ROOT / "examples/data/physical-grid"
    rcte_manifest = json.loads((physical_root / "rcte/manifest.json").read_text())
    photo_manifest = json.loads((physical_root / "photochemical/manifest.json").read_text())
    with np.load(physical_root / "rcte/grid.npz", allow_pickle=False) as archive:
        rcte = {key: archive[key].copy() for key in archive.files}
    with np.load(physical_root / "photochemical/grid.npz", allow_pickle=False) as archive:
        photo = {key: archive[key].copy() for key in archive.files}
    wave = (rcte["edges"][:-1] + rcte["edges"][1:]) / 2
    labels = np.array(
        [state["log_metallicity"] for state in rcte_manifest["states"]], dtype=np.float64
    )
    train = np.array(rcte_manifest["splits"]["train"], dtype=np.int64)
    interior_calibration = [
        index for index in rcte_manifest["splits"]["validation"] if -1 < labels[index] < 2
    ]
    interior_test = [index for index in rcte_manifest["splits"]["test"] if -1 < labels[index] < 2]
    calibration_indices = np.array(interior_calibration[:12], dtype=np.int64)
    validation_indices = np.array(interior_test[:12], dtype=np.int64)
    observation_index = interior_test[12]
    rng = np.random.default_rng(20261014)
    calibration_observations = rcte["spectra"][calibration_indices, 0] + rng.normal(
        0, rcte["sigma"], size=(len(calibration_indices), len(wave))
    )
    validation_observations = rcte["spectra"][validation_indices, 0] + rng.normal(
        0, rcte["sigma"], size=(len(validation_indices), len(wave))
    )
    observed = rcte["spectra"][observation_index, 0] + rng.normal(0, rcte["sigma"])

    source = "doi:10.5281/zenodo.7236759; exploratory exposed PICASO RCTE subsets"
    support = _write_bundle(
        output,
        "support",
        "rcte-support",
        source,
        [rcte_manifest["states"][index] for index in train],
        {
            "wavelength_microns": wave,
            "spectra": rcte["spectra"][train],
            "labels": labels[train],
        },
    )
    calibration = _write_bundle(
        output,
        "calibration",
        "rcte-calibration",
        source,
        [rcte_manifest["states"][index] for index in calibration_indices],
        {
            "wavelength_microns": wave,
            "observations": calibration_observations,
            "sigma": rcte["sigma"],
            "labels": labels[calibration_indices],
        },
    )
    validation = _write_bundle(
        output,
        "validation",
        "rcte-validation",
        source,
        [rcte_manifest["states"][index] for index in validation_indices],
        {
            "wavelength_microns": wave,
            "observations": validation_observations,
            "sigma": rcte["sigma"],
            "labels": labels[validation_indices],
        },
    )
    observation = _write_bundle(
        output,
        "observation",
        "rcte-observation",
        source,
        [rcte_manifest["states"][observation_index]],
        {"wavelength_microns": wave, "observation": observed, "sigma": rcte["sigma"]},
    )
    interior_report = run_reliability_audit(support, calibration, validation, observation)
    interior_report["demonstration_context"] = {
        "status": "outcome-informed software-path example only",
        "restriction": "calibration and validation use interior labels {0.0, 0.5, 1.0}",
        "research_use": "invalid for general RCTE qualification",
    }
    write_audit_report(interior_report, output / "interior-label-report.json")

    photo_wave = (photo["edges"][:-1] + photo["edges"][1:]) / 2
    wrong_family = _write_bundle(
        output,
        "observation",
        "photochemical-observation",
        "doi:10.5281/zenodo.7236759; exposed photochemical family",
        [photo_manifest["states"][0]],
        {
            "wavelength_microns": photo_wave,
            "observation": photo["spectra"][0, 0],
            "sigma": photo["sigma"],
        },
    )
    wrong_report = run_reliability_audit(support, calibration, validation, wrong_family)
    wrong_report["demonstration_context"] = {
        "status": "exposed wrong-family control under outcome-informed interior qualification",
        "observation": "prespecified photochemical state index 0",
    }
    write_audit_report(wrong_report, output / "wrong-family-report.json")

    full_calibration_indices = np.array(rcte_manifest["splits"]["validation"], dtype=np.int64)
    full_validation_indices = np.array(rcte_manifest["splits"]["test"], dtype=np.int64)
    full_rng = np.random.default_rng(np.random.SeedSequence([20261014, 1]))
    full_calibration_observations = rcte["spectra"][full_calibration_indices, 0] + full_rng.normal(
        0, rcte["sigma"], size=(len(full_calibration_indices), len(wave))
    )
    full_validation_observations = rcte["spectra"][full_validation_indices, 0] + full_rng.normal(
        0, rcte["sigma"], size=(len(full_validation_indices), len(wave))
    )
    full_calibration = _write_bundle(
        output,
        "calibration",
        "rcte-full-calibration",
        source,
        [rcte_manifest["states"][index] for index in full_calibration_indices],
        {
            "wavelength_microns": wave,
            "observations": full_calibration_observations,
            "sigma": rcte["sigma"],
            "labels": labels[full_calibration_indices],
        },
    )
    full_validation = _write_bundle(
        output,
        "validation",
        "rcte-full-validation",
        source,
        [rcte_manifest["states"][index] for index in full_validation_indices],
        {
            "wavelength_microns": wave,
            "observations": full_validation_observations,
            "sigma": rcte["sigma"],
            "labels": labels[full_validation_indices],
        },
    )
    full_report = run_reliability_audit(support, full_calibration, full_validation, wrong_family)
    full_report["demonstration_context"] = {
        "status": "lead exploratory evidence under the corrected operational rule",
        "calibration": "all 32 exposed RCTE validation states",
        "validation": "all 32 exposed RCTE test states",
        "observation": "prespecified exposed photochemical state index 0",
        "fresh_science": False,
    }
    write_audit_report(full_report, output / "full-heldout-report.json")

    mismatch_state = dict(photo_manifest["states"][1])
    mismatch_state["id"] = str(mismatch_state["id"]) + ":shifted-geometry-control"
    mismatch_state["sha256"] = _fingerprint([str(mismatch_state["sha256"]), "geometry-shift"])
    geometry_mismatch = _write_bundle(
        output,
        "observation",
        "geometry-mismatch-observation",
        "adversarial coordinate mismatch control",
        [mismatch_state],
        {
            "wavelength_microns": photo_wave + 1e-6,
            "observation": photo["spectra"][1, 0],
            "sigma": photo["sigma"],
        },
    )
    try:
        run_reliability_audit(support, calibration, validation, geometry_mismatch)
    except ValueError as error:
        geometry_result = {"rejected": True, "reason": str(error)}
    else:  # pragma: no cover - an executable adversarial assertion
        raise AssertionError("Geometry mismatch was not rejected")
    (output / "controls.json").write_text(
        json.dumps(
            {
                "claim_boundary": "Exploratory software demonstration on exposed states.",
                "full_heldout_decision": full_report["decision"],
                "full_heldout_qualification": full_report["score_calibration"]["qualified"],
                "wrong_family_decision": wrong_report["decision"],
                "interior_label_software_path_decision": interior_report["decision"],
                "geometry_mismatch": geometry_result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "demo-design.json").write_text(
        json.dumps(
            {
                "status": "outcome-informed software demonstration; not scientific evidence",
                "timeline": [
                    (
                        "An earlier demo ran before support-boundary intervals were included "
                        "in the operational validation rule."
                    ),
                    (
                        "Independent review required validation and observation to use the "
                        "same boundary abstention."
                    ),
                    (
                        "After that outcome, the accept-path panel was restricted to interior "
                        "labels and is labelled accordingly."
                    ),
                    (
                        "The full 32-state calibration and 32-state validation panel was then "
                        "added without changing gates."
                    ),
                ],
                "interior_label_panel": {
                    "calibration_labels": sorted(
                        set(float(value) for value in labels[calibration_indices])
                    ),
                    "validation_labels": sorted(
                        set(float(value) for value in labels[validation_indices])
                    ),
                    "observation_label_used_only_to_construct_demo": float(
                        labels[observation_index]
                    ),
                },
                "full_panel": {
                    "calibration_states": len(full_calibration_indices),
                    "validation_states": len(full_validation_indices),
                    "observation": "prespecified exposed photochemical state index 0",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "full_heldout": full_report["decision"],
                "interior_software_example": interior_report["decision"],
                "photochemical_observation": wrong_report["decision"],
                "geometry_mismatch": "rejected",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
