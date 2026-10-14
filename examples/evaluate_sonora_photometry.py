"""Run the frozen Sonora photometry qualification and conditional cloud challenge once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from tierra_plus.diagnostics.physical_grid import FiniteSpectralGrid
from tierra_plus.provenance import hash_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples/configs/sonora-diamondback-photometry-v1.json"
DEFAULT_PREPARATION = ROOT / "examples/results/sonora-photometry-preparation/record.json"
MAX_JSON_BYTES = 1_000_000
MAX_GRID_BYTES = 16_000_000
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EXPECTED_ARRAYS = {
    "feature_names",
    "magnitudes",
    "effective_temperature_k",
    "gravity_m_s2",
    "logg_cgs",
    "metallicity_dex",
    "fsed",
    "age_gyr",
    "mass_jupiter",
    "radius_jupiter",
    "log_luminosity_solar",
    "state_ids",
    "state_sha256",
    "support_indices",
    "calibration_indices",
    "validation_indices",
    "cloud_challenge_indices",
}


def _read_regular(path: Path, maximum: int, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file, not a symbolic link")
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
    except OSError as error:
        raise ValueError(f"{name} could not be read") from error
    if len(payload) > maximum:
        raise ValueError(f"{name} exceeds its size bound")
    return payload


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    payload = _read_regular(path, MAX_JSON_BYTES, name)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value), hash_bytes(payload)


def load_inputs(
    dataset: Path, config_path: Path, preparation_path: Path
) -> tuple[dict[str, NDArray[Any]], dict[str, Any], dict[str, str]]:
    """Load only the exact prepared grid bound by the frozen public record."""
    if dataset.is_symlink() or not dataset.is_dir():
        raise ValueError("Sonora dataset must be a regular directory, not a symbolic link")
    config, config_sha256 = _read_json(config_path, "Sonora protocol")
    preparation, preparation_sha256 = _read_json(preparation_path, "Sonora preparation record")
    manifest, manifest_sha256 = _read_json(dataset / "manifest.json", "Sonora manifest")
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_before_inference"
        or preparation.get("status") != "prepared_without_inference"
        or preparation.get("inference_run") is not False
        or manifest.get("status") != "prepared_without_inference"
    ):
        raise ValueError("Sonora protocol and inputs must retain their pre-inference statuses")
    if preparation.get("protocol_sha256") != config_sha256:
        raise ValueError("Sonora protocol differs from the public frozen preparation record")
    if manifest.get("protocol_sha256") != config_sha256:
        raise ValueError("Sonora manifest was prepared from a different protocol")

    grid_payload = _read_regular(dataset / "grid.npz", MAX_GRID_BYTES, "Sonora grid")
    grid_sha256 = hash_bytes(grid_payload)
    if (
        preparation.get("canonical_grid_sha256") != grid_sha256
        or manifest.get("grid_sha256") != grid_sha256
    ):
        raise ValueError("Sonora grid differs from the frozen prepared artifact")
    try:
        with zipfile.ZipFile(BytesIO(grid_payload)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected_names = {f"{name}.npy" for name in EXPECTED_ARRAYS}
            if (
                set(names) != expected_names
                or len(names) != len(set(names))
                or any(
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    for member in members
                )
            ):
                raise ValueError("Sonora grid members differ from the canonical schema")
        with np.load(BytesIO(grid_payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in EXPECTED_ARRAYS}
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("Sonora grid is not a valid canonical NPZ archive") from error
    _validate_arrays(arrays, config)
    return (
        arrays,
        config,
        {
            "protocol_sha256": config_sha256,
            "preparation_record_sha256": preparation_sha256,
            "manifest_sha256": manifest_sha256,
            "canonical_grid_sha256": grid_sha256,
        },
    )


def _role_indices(arrays: dict[str, NDArray[Any]], role: str) -> IntArray:
    values = np.asarray(arrays[f"{role}_indices"])
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"Sonora {role} indices must be a one-dimensional integer array")
    return values.astype(np.int64, copy=False)


def _validate_arrays(arrays: dict[str, NDArray[Any]], config: dict[str, Any]) -> None:
    if set(arrays) != EXPECTED_ARRAYS:
        raise ValueError("Sonora grid arrays differ from the canonical schema")
    rows = int(cast(dict[str, Any], config["expected_grid"])["rows"])
    features = [str(value) for value in cast(list[str], config["photometric_features"])]
    magnitudes = np.asarray(arrays["magnitudes"])
    numeric_rows = (
        "effective_temperature_k",
        "gravity_m_s2",
        "logg_cgs",
        "metallicity_dex",
        "fsed",
        "age_gyr",
        "mass_jupiter",
        "radius_jupiter",
        "log_luminosity_solar",
    )
    if magnitudes.shape != (rows, len(features)) or magnitudes.dtype.kind not in "fiu":
        raise ValueError("Sonora magnitude matrix has an unexpected shape or dtype")
    if any(
        np.asarray(arrays[name]).shape != (rows,)
        or np.asarray(arrays[name]).dtype.kind not in "fiu"
        for name in numeric_rows
    ):
        raise ValueError("Sonora physical coordinate arrays have unexpected shapes or dtypes")
    if not np.isfinite(magnitudes).all() or any(
        not np.isfinite(np.asarray(arrays[name], dtype=np.float64)).all() for name in numeric_rows
    ):
        raise ValueError("Sonora grid contains non-finite numeric values")
    if np.asarray(arrays["feature_names"]).tolist() != features:
        raise ValueError("Sonora feature order differs from the frozen protocol")
    for name in ("state_ids", "state_sha256"):
        value = np.asarray(arrays[name])
        if value.shape != (rows,) or value.dtype.kind != "U" or len(set(value.tolist())) != rows:
            raise ValueError(f"Sonora {name} must contain unique Unicode identities")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in np.asarray(arrays["state_sha256"]).tolist()
    ):
        raise ValueError("Sonora state_sha256 values must be lowercase SHA-256 digests")

    partition = cast(dict[str, dict[str, Any]], config["partition"])
    gravity = np.asarray(arrays["gravity_m_s2"], dtype=np.float64)
    fsed = np.asarray(arrays["fsed"], dtype=np.int64)
    roles = ("support", "calibration", "validation", "cloud_challenge")
    seen: dict[str, set[int]] = {}
    for role in roles:
        indices = _role_indices(arrays, role)
        if (
            len(indices) != int(partition[role]["expected_states"])
            or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= rows)
        ):
            raise ValueError(f"Sonora {role} indices violate the frozen count or bounds")
        expected = np.flatnonzero(
            np.isin(fsed, np.asarray(partition[role]["fsed"], dtype=np.int64))
            & np.isin(gravity, np.asarray(partition[role]["gravity_m_s2"], dtype=np.float64))
        )
        if not np.array_equal(indices, expected):
            raise ValueError(f"Sonora {role} indices differ from the frozen coordinate rule")
        seen[role] = set(indices.tolist())
    for left, right in (
        ("support", "calibration"),
        ("support", "validation"),
        ("calibration", "validation"),
    ):
        if seen[left] & seen[right]:
            raise ValueError("Sonora cloud-free partitions overlap")


def build_support_grid(
    arrays: dict[str, NDArray[Any]], config: dict[str, Any], sigma: FloatArray
) -> FiniteSpectralGrid:
    """Arrange support as temperature x (gravity, metallicity) x band."""
    expected_grid = cast(dict[str, Any], config["expected_grid"])
    partition = cast(dict[str, dict[str, Any]], config["partition"])
    temperatures = np.asarray(expected_grid["temperatures_k"], dtype=np.float64)
    gravities = np.asarray(partition["support"]["gravity_m_s2"], dtype=np.float64)
    metallicities = np.asarray(expected_grid["metallicity_dex"], dtype=np.float64)
    magnitudes = np.asarray(arrays["magnitudes"], dtype=np.float64)
    temperature = np.asarray(arrays["effective_temperature_k"], dtype=np.float64)
    gravity = np.asarray(arrays["gravity_m_s2"], dtype=np.float64)
    metallicity = np.asarray(arrays["metallicity_dex"], dtype=np.float64)
    fsed = np.asarray(arrays["fsed"], dtype=np.int64)
    cube = np.empty((len(temperatures), len(gravities) * len(metallicities), len(sigma)))
    for temperature_index, target in enumerate(temperatures):
        nuisance_index = 0
        for nuisance_gravity in gravities:
            for nuisance_metallicity in metallicities:
                match = np.flatnonzero(
                    (temperature == target)
                    & (gravity == nuisance_gravity)
                    & (metallicity == nuisance_metallicity)
                    & (fsed == 0)
                )
                if len(match) != 1:
                    raise ValueError("Sonora support does not contain one state per nuisance cell")
                cube[temperature_index, nuisance_index] = magnitudes[match[0]]
                nuisance_index += 1
    return FiniteSpectralGrid(cube, temperatures, sigma)


def _infer_checked(
    grid: FiniteSpectralGrid, observations: FloatArray, role: str
) -> dict[str, FloatArray]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            result = grid.infer(observations)
    except FloatingPointError as error:
        raise ValueError(f"Sonora {role} inference is numerically invalid") from error
    expected = {"mean", "lower", "upper", "envelope_lower", "envelope_upper", "chi2"}
    if set(result) != expected or any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"Sonora {role} inference produced non-finite outputs")
    return result


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    radius = (
        z / denominator * math.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2))
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def selection_metrics(
    result: dict[str, FloatArray], truth: FloatArray, threshold: float, width_cap: float
) -> dict[str, Any]:
    """Apply exactly the frozen score and central-interval-width selection rule."""
    if not math.isfinite(threshold) or not math.isfinite(width_cap) or width_cap <= 0:
        raise ValueError("Sonora threshold must be finite and width cap must be positive finite")
    try:
        with np.errstate(over="raise", invalid="raise"):
            width = result["upper"] - result["lower"]
    except FloatingPointError as error:
        raise ValueError("Sonora interval width is numerically invalid") from error
    if (
        truth.ndim != 1
        or any(value.shape != truth.shape for value in result.values())
        or not np.isfinite(truth).all()
        or not np.isfinite(width).all()
        or np.any(width < 0)
    ):
        raise ValueError("Sonora inference outputs and truth have inconsistent finite shapes")
    score_ok = result["chi2"] <= threshold
    width_ok = width <= width_cap
    accepted = score_ok & width_ok
    covered = (truth >= result["lower"]) & (truth <= result["upper"])
    accepted_count = int(accepted.sum())
    covered_count = int(covered.sum())
    accepted_covered = int((accepted & covered).sum())
    return {
        "states": int(truth.size),
        "accepted": accepted_count,
        "acceptance": float(accepted.mean()),
        "acceptance_wilson95": _wilson(accepted_count, int(truth.size)),
        "covered": covered_count,
        "coverage": float(covered.mean()),
        "coverage_wilson95": _wilson(covered_count, int(truth.size)),
        "accepted_covered": accepted_covered,
        "accepted_coverage": (float(accepted_covered / accepted_count) if accepted_count else None),
        "accepted_coverage_wilson95": _wilson(accepted_covered, accepted_count),
        "score_accepted": int(score_ok.sum()),
        "width_accepted": int(width_ok.sum()),
        "median_interval_width_k": float(np.median(width)),
        "maximum_interval_width_k": float(width.max()),
    }


def _gate_failures(metrics: dict[str, Any], planned: dict[str, Any], prefix: str) -> list[str]:
    failures: list[str] = []
    if float(metrics["acceptance"]) < float(planned["minimum_acceptance"]):
        failures.append(f"{prefix}_acceptance_below_floor")
    accepted_coverage = metrics["accepted_coverage"]
    if accepted_coverage is None or float(accepted_coverage) < float(
        planned["minimum_accepted_coverage"]
    ):
        failures.append(f"{prefix}_accepted_coverage_below_floor")
    return failures


def evaluate(
    arrays: dict[str, NDArray[Any]], config: dict[str, Any], identities: dict[str, str]
) -> dict[str, Any]:
    """Evaluate validation first and infer the cloudy challenge only after qualification."""
    planned = cast(dict[str, Any], config["planned_evaluation"])
    if planned.get("status") != "not_run":
        raise ValueError("Sonora planned evaluation no longer has its frozen not-run status")
    sigma_value = float(planned["independent_sigma_magnitude"])
    if not math.isfinite(sigma_value) or sigma_value <= 0:
        raise ValueError("Sonora noise scale must be positive and finite")
    magnitudes = np.asarray(arrays["magnitudes"], dtype=np.float64)
    sigma = np.full(magnitudes.shape[1], sigma_value, dtype=np.float64)
    grid = build_support_grid(arrays, config, sigma)
    seed = [int(value) for value in cast(list[int], planned["noise_seed"])]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    noise = generator.standard_normal(magnitudes.shape, dtype=np.float64) * sigma_value
    if not np.isfinite(noise).all():
        raise ValueError("Sonora fixed noise realization is non-finite")
    observations = magnitudes + noise
    truth = np.asarray(arrays["effective_temperature_k"], dtype=np.float64)
    calibration_indices = _role_indices(arrays, "calibration")
    validation_indices = _role_indices(arrays, "validation")
    challenge_indices = _role_indices(arrays, "cloud_challenge")

    calibration = _infer_checked(grid, observations[calibration_indices], "calibration")
    score_quantile = float(planned["score_quantile"])
    threshold = float(np.quantile(calibration["chi2"], score_quantile, method="higher"))
    if not math.isfinite(threshold):
        raise ValueError("Sonora calibration produced a non-finite score threshold")
    width_cap = float(planned["maximum_temperature_interval_width_k"])
    calibration_metrics = selection_metrics(
        calibration, truth[calibration_indices], threshold, width_cap
    )
    validation = _infer_checked(grid, observations[validation_indices], "validation")
    validation_metrics = selection_metrics(
        validation, truth[validation_indices], threshold, width_cap
    )
    validation_failures = _gate_failures(validation_metrics, planned, "validation")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "validation_failed_challenge_not_run" if validation_failures else "running",
        "verdict": "validation_failed" if validation_failures else "pending_cloud_challenge",
        "claim_boundary": config["claim_boundary"],
        "question": config["question"],
        "identities": identities,
        "noise": {
            "seed_sequence": seed,
            "bit_generator": "PCG64",
            "independent_sigma_magnitude": sigma_value,
            "canonical_full_grid_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        },
        "rule": {
            "likelihood": planned["likelihood"],
            "prior": (
                "uniform over 16 temperatures and 9 cloud-free "
                "gravity/metallicity nuisance states"
            ),
            "interval": "central 90% marginal discrete-temperature posterior interval",
            "acceptance": (
                "minimum_chi2 <= calibration_threshold AND "
                "central_90_interval_width_k <= maximum_temperature_interval_width_k"
            ),
            "score_quantile": score_quantile,
            "minimum_acceptance": float(planned["minimum_acceptance"]),
            "minimum_accepted_coverage": float(planned["minimum_accepted_coverage"]),
            "maximum_temperature_interval_width_k": width_cap,
        },
        "score_calibration": {
            "threshold": threshold,
            "metrics": calibration_metrics,
        },
        "validation": {
            "qualified": not validation_failures,
            "failures": validation_failures,
            "metrics": validation_metrics,
        },
        "cloud_challenge": {
            "status": "not_run_validation_failed" if validation_failures else "pending"
        },
        "small_sample_note": (
            "Wilson intervals summarize finite counts; gates use frozen point estimates and do "
            "not establish reliability outside this synthetic photometry grid and noise model."
        ),
    }
    if validation_failures:
        return report

    challenge = _infer_checked(grid, observations[challenge_indices], "cloud_challenge")
    challenge_metrics = selection_metrics(challenge, truth[challenge_indices], threshold, width_cap)
    challenge_failures = _gate_failures(challenge_metrics, planned, "cloud_challenge")
    report["status"] = "completed"
    report["verdict"] = "cloud_challenge_failed" if challenge_failures else "cloud_challenge_passed"
    report["cloud_challenge"] = {
        "status": "completed",
        "passed": not challenge_failures,
        "failures": challenge_failures,
        "metrics": challenge_metrics,
        "paired_validation_states": int(len(validation_indices)),
        "cloud_states_per_validation_state": int(len(challenge_indices) // len(validation_indices)),
    }
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(f"Refusing to replace existing Sonora result: {output}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preparation-record", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    arrays, config, identities = load_inputs(args.dataset, args.config, args.preparation_record)
    identities["evaluation_harness_sha256"] = hash_bytes(Path(__file__).read_bytes())
    report = evaluate(arrays, config, identities)
    write_report(report, args.output)
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
