"""Deployable finite-grid inference with an empirical abstention contract.

The audit is conditional on the supplied spectra, labels, nuisance support and
independent-noise likelihood.  Its decision is not a generic out-of-distribution
guarantee.  Calibration selects a score threshold; a disjoint labelled validation
set decides whether that threshold is usable; an unlabelled observation is never
used to tune either quantity.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from tierra_plus import __version__
from tierra_plus.diagnostics.physical_grid import FiniteSpectralGrid
from tierra_plus.provenance import hash_bytes

FloatArray = NDArray[np.float64]
MAX_MANIFEST_BYTES = 1_000_000
MAX_ARCHIVE_BYTES = 128_000_000
MAX_MEMBER_BYTES = 96_000_000
MAX_ARCHIVE_MEMBERS = 8
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AuditCriteria:
    """Predeclared empirical gates for one reliability audit."""

    score_quantile: float = 0.9
    minimum_states: int = 10
    minimum_acceptance: float = 0.7
    minimum_accepted_coverage: float = 0.85
    maximum_interval_width: float = 0.6

    def __post_init__(self) -> None:
        if not 0 < self.score_quantile < 1:
            raise ValueError("score_quantile must lie in (0, 1)")
        if type(self.minimum_states) is not int or self.minimum_states < 2:
            raise ValueError("minimum_states must be an integer of at least two")
        for name, value in (
            ("minimum_acceptance", self.minimum_acceptance),
            ("minimum_accepted_coverage", self.minimum_accepted_coverage),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.maximum_interval_width) or self.maximum_interval_width <= 0:
            raise ValueError("maximum_interval_width must be positive and finite")


@dataclass(frozen=True)
class _Bundle:
    role: str
    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, FloatArray]
    archive_sha256: str
    manifest_sha256: str


def _manifest(bundle: Path, expected_role: str) -> tuple[dict[str, Any], str]:
    manifest_path = bundle / "manifest.json"
    if bundle.is_symlink() or manifest_path.is_symlink():
        raise ValueError(f"{expected_role} bundle and manifest must not be symbolic links")
    if not bundle.is_dir() or not manifest_path.is_file():
        raise ValueError(f"{expected_role} bundle needs a regular manifest.json")
    try:
        with manifest_path.open("rb") as handle:
            payload = handle.read(MAX_MANIFEST_BYTES + 1)
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ValueError(f"{expected_role} manifest exceeds the size limit")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{expected_role} manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{expected_role} manifest must be a JSON object")
    result = cast(dict[str, Any], value)
    required = {
        "schema_version",
        "role",
        "dataset_id",
        "source",
        "source_fingerprint",
        "state_ids",
        "state_fingerprints",
        "data_file",
        "data_sha256",
        "wavelength_unit",
        "spectral_quantity",
    }
    if not required.issubset(result):
        missing = sorted(required - result.keys())
        raise ValueError(f"{expected_role} manifest is missing: {', '.join(missing)}")
    if result["schema_version"] != 1 or result["role"] != expected_role:
        raise ValueError(f"Expected a schema-1 {expected_role} bundle")
    if result["data_file"] != "data.npz":
        raise ValueError("Bundle data_file must be exactly data.npz")
    if result["wavelength_unit"] != "micron":
        raise ValueError("Only wavelength_unit=micron is supported")
    if result["spectral_quantity"] != "transit_depth":
        raise ValueError("Only spectral_quantity=transit_depth is supported")
    for key in ("dataset_id", "source"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{expected_role} {key} must be a nonempty string")
    for key in ("source_fingerprint", "data_sha256"):
        if not isinstance(result[key], str) or _SHA256.fullmatch(result[key]) is None:
            raise ValueError(f"{expected_role} {key} must be a lowercase SHA-256 digest")
    for key in ("state_ids", "state_fingerprints"):
        items = result[key]
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
            or len(set(items)) != len(items)
        ):
            raise ValueError(f"{expected_role} {key} must contain unique nonempty strings")
        if key == "state_fingerprints" and any(_SHA256.fullmatch(item) is None for item in items):
            raise ValueError(
                f"{expected_role} state_fingerprints must be lowercase SHA-256 digests"
            )
    if len(result["state_ids"]) != len(result["state_fingerprints"]):
        raise ValueError(f"{expected_role} state identifiers and fingerprints differ in count")
    return result, hash_bytes(payload)


def _preflight_npy(archive: zipfile.ZipFile, member: zipfile.ZipInfo, role: str) -> int:
    try:
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError("unsupported NPY header version")
            header_bytes = handle.tell()
    except (EOFError, OSError, ValueError) as error:
        raise ValueError(f"{role} archive has an invalid NPY header") from error
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "fiu":
        raise ValueError(f"{role} arrays must use real numeric, non-object dtypes")
    elements = math.prod(shape)
    declared_bytes = elements * dtype.itemsize
    expanded_bytes = elements * np.dtype(np.float64).itemsize
    if declared_bytes != member.file_size - header_bytes:
        raise ValueError(f"{role} NPY shape does not match its bounded member payload")
    if declared_bytes > MAX_MEMBER_BYTES or expanded_bytes > MAX_MEMBER_BYTES:
        raise ValueError(f"{role} NPY array exceeds the in-memory size limit")
    return expanded_bytes


def _archive_arrays(payload: bytes, role: str) -> dict[str, FloatArray]:
    expected = {
        "support": {"wavelength_microns", "spectra", "labels"},
        "calibration": {"wavelength_microns", "observations", "sigma", "labels"},
        "validation": {"wavelength_microns", "observations", "sigma", "labels"},
        "observation": {"wavelength_microns", "observation", "sigma"},
    }[role]
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            names = {f"{key}.npy" for key in expected}
            member_names = [item.filename for item in members]
            if (
                len(members) > MAX_ARCHIVE_MEMBERS
                or len(members) != len(names)
                or set(member_names) != names
                or len(set(member_names)) != len(member_names)
            ):
                raise ValueError(f"{role} archive members must be exactly {sorted(names)}")
            if any(
                item.is_dir()
                or item.flag_bits & 0x1
                or item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or item.file_size > MAX_MEMBER_BYTES
                or item.file_size < 1
                or item.compress_size < 1
                or item.file_size / item.compress_size > 1000
                for item in members
            ):
                raise ValueError(f"{role} archive member exceeds safety bounds")
            if sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
                raise ValueError(f"{role} archive exceeds the uncompressed size limit")
            expanded_total = sum(_preflight_npy(archive, member, role) for member in members)
            if expanded_total > MAX_ARCHIVE_BYTES:
                raise ValueError(f"{role} arrays exceed the total in-memory size limit")
    except (NotImplementedError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(f"{role} data.npz is not a valid bounded NPZ archive") from error
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            result = {key: np.asarray(archive[key], dtype=np.float64).copy() for key in expected}
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise ValueError(f"{role} archive contains invalid NumPy arrays") from error
    if any(not value.size or not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{role} arrays must be nonempty and finite")
    return result


def _load_bundle(path: Path, role: str) -> _Bundle:
    if path.is_symlink():
        raise ValueError(f"{role} bundle must not be a symbolic link")
    bundle = path.resolve(strict=True)
    manifest, manifest_hash = _manifest(bundle, role)
    archive_path = bundle / "data.npz"
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError(f"{role} data.npz must be a regular file, not a symbolic link")
    size = archive_path.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{role} archive exceeds the compressed size limit")
    try:
        with archive_path.open("rb") as handle:
            payload = handle.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as error:
        raise ValueError(f"{role} data.npz could not be read") from error
    if len(payload) != size or len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{role} data.npz changed or exceeded bounds while being read")
    actual_hash = hash_bytes(payload)
    if actual_hash != manifest["data_sha256"]:
        raise ValueError(f"{role} data.npz does not match its declared SHA-256")
    arrays = _archive_arrays(payload, role)
    expected_rows = len(manifest["state_ids"])
    row_key = "spectra" if role == "support" else "observation"
    if role in {"calibration", "validation"}:
        row_key = "observations"
    if arrays[row_key].ndim == 0:
        raise ValueError(f"{role} primary array must not be scalar")
    rows = 1 if arrays[row_key].ndim == 1 else arrays[row_key].shape[0]
    if rows != expected_rows:
        raise ValueError(f"{role} state metadata count does not match its arrays")
    return _Bundle(role, bundle, manifest, arrays, actual_hash, manifest_hash)


def _validate_independence(bundles: list[_Bundle]) -> None:
    dataset_ids = [str(bundle.manifest["dataset_id"]) for bundle in bundles]
    source_fingerprints = [str(bundle.manifest["source_fingerprint"]) for bundle in bundles]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("All audit roles need distinct dataset_id values")
    if len(set(source_fingerprints)) != len(source_fingerprints):
        raise ValueError("All audit roles need distinct declared source fingerprints")
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for bundle in bundles:
        ids = set(cast(list[str], bundle.manifest["state_ids"]))
        fingerprints = set(cast(list[str], bundle.manifest["state_fingerprints"]))
        if seen_ids & ids:
            raise ValueError("A declared physical state_id is reused across audit roles")
        if seen_fingerprints & fingerprints:
            raise ValueError("A declared physical-state fingerprint is reused across audit roles")
        seen_ids |= ids
        seen_fingerprints |= fingerprints


def _geometry(bundle: _Bundle, reference: FloatArray) -> None:
    wave = bundle.arrays["wavelength_microns"]
    if wave.ndim != 1 or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError(f"{bundle.role} wavelengths must be positive, one-dimensional bin centers")
    if not np.array_equal(wave, reference):
        raise ValueError(f"{bundle.role} wavelength centers differ from support exactly")
    if bundle.role == "support":
        spectra = bundle.arrays["spectra"]
        labels = bundle.arrays["labels"]
        if spectra.ndim not in {2, 3} or spectra.shape[0] != labels.size:
            raise ValueError("Support spectra and labels have inconsistent shapes")
        if spectra.shape[-1] != wave.size or labels.ndim != 1:
            raise ValueError("Support arrays do not match the wavelength geometry")
    elif bundle.role in {"calibration", "validation"}:
        observations = bundle.arrays["observations"]
        sigma = bundle.arrays["sigma"]
        labels = bundle.arrays["labels"]
        if observations.ndim != 2 or observations.shape != (labels.size, wave.size):
            raise ValueError(f"{bundle.role} observations and labels have inconsistent shapes")
        if sigma.shape != wave.shape or np.any(sigma <= 0) or labels.ndim != 1:
            raise ValueError(f"{bundle.role} needs one positive uncertainty per wavelength bin")
    else:
        observation = bundle.arrays["observation"]
        sigma = bundle.arrays["sigma"]
        if observation.shape != wave.shape or sigma.shape != wave.shape or np.any(sigma <= 0):
            raise ValueError(
                "Observation depth and positive uncertainty must match each bin center"
            )


def _grid(support: _Bundle, sigma: FloatArray) -> FiniteSpectralGrid:
    spectra = support.arrays["spectra"]
    if spectra.ndim == 2:
        spectra = spectra[:, None, :]
    return FiniteSpectralGrid(spectra, support.arrays["labels"], sigma)


def _infer_checked(
    grid: FiniteSpectralGrid, observations: FloatArray, role: str
) -> dict[str, FloatArray]:
    """Run finite-grid inference and reject every non-finite numeric outcome."""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            result = grid.infer(observations)
    except FloatingPointError as error:
        raise ValueError(f"{role} inference is numerically invalid") from error
    expected = {"mean", "lower", "upper", "envelope_lower", "envelope_upper", "chi2"}
    if set(result) != expected or any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{role} inference produced non-finite outputs")
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


def _label_support(truth: FloatArray, support_labels: FloatArray, role: str) -> None:
    if any(not np.isclose(value, support_labels, atol=1e-12, rtol=1e-12).any() for value in truth):
        raise ValueError(f"{role} contains labels outside the discrete support")


def _selection_metrics(
    result: dict[str, FloatArray],
    truth: FloatArray,
    threshold: float,
    width_cap: float,
    support_min: float,
    support_max: float,
) -> dict[str, Any]:
    try:
        with np.errstate(over="raise", invalid="raise"):
            width = result["upper"] - result["lower"]
    except FloatingPointError as error:
        raise ValueError("Inference interval width is numerically invalid") from error
    if not np.isfinite(width).all():
        raise ValueError("Inference interval width is non-finite")
    score_ok = result["chi2"] <= threshold
    width_ok = width <= width_cap
    boundary_ok = (result["lower"] > support_min) & (result["upper"] < support_max)
    accepted = score_ok & width_ok & boundary_ok
    covered = (truth >= result["lower"]) & (truth <= result["upper"])
    accepted_count = int(accepted.sum())
    accepted_covered = int((accepted & covered).sum())
    return {
        "states": int(truth.size),
        "accepted": accepted_count,
        "acceptance": float(accepted.mean()),
        "acceptance_wilson95": _wilson(accepted_count, int(truth.size)),
        "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "coverage_wilson95": _wilson(int(covered.sum()), int(truth.size)),
        "accepted_covered": accepted_covered,
        "accepted_coverage": (float(accepted_covered / accepted_count) if accepted_count else None),
        "accepted_coverage_wilson95": _wilson(accepted_covered, accepted_count),
        "score_accepted": int(score_ok.sum()),
        "width_accepted": int(width_ok.sum()),
        "boundary_accepted": int(boundary_ok.sum()),
        "median_interval_width": float(np.median(width)),
        "maximum_interval_width": float(width.max()),
    }


def run_reliability_audit(
    support_path: Path,
    calibration_path: Path,
    validation_path: Path,
    observation_path: Path,
    criteria: AuditCriteria | None = None,
) -> dict[str, Any]:
    """Audit one spectrum against finite support and disjoint empirical controls.

    Bundle independence is verified only from identifiers and fingerprints supplied
    by the producer.  Hashes establish byte identity, not the truth or independence
    of metadata claims.
    """
    if criteria is None:
        criteria = AuditCriteria()
    support = _load_bundle(support_path, "support")
    calibration = _load_bundle(calibration_path, "calibration")
    validation = _load_bundle(validation_path, "validation")
    observation = _load_bundle(observation_path, "observation")
    bundles = [support, calibration, validation, observation]
    _validate_independence(bundles)
    support_wave = support.arrays["wavelength_microns"]
    for bundle in bundles:
        _geometry(bundle, support_wave)
    reference_sigma = calibration.arrays["sigma"]
    for bundle in (validation, observation):
        if not np.array_equal(bundle.arrays["sigma"], reference_sigma):
            raise ValueError(f"{bundle.role} uncertainty geometry differs from calibration exactly")
    for bundle in (calibration, validation):
        states = len(bundle.manifest["state_ids"])
        if states < criteria.minimum_states:
            raise ValueError(
                f"{bundle.role} needs at least {criteria.minimum_states} unique physical states"
            )
        _label_support(bundle.arrays["labels"], support.arrays["labels"], bundle.role)

    calibration_grid = _grid(support, calibration.arrays["sigma"])
    calibration_result = _infer_checked(
        calibration_grid, calibration.arrays["observations"], "calibration"
    )
    threshold = float(
        np.quantile(calibration_result["chi2"], criteria.score_quantile, method="higher")
    )
    if not math.isfinite(threshold):
        raise ValueError("Calibration produced a non-finite score threshold")
    support_min = float(support.arrays["labels"].min())
    support_max = float(support.arrays["labels"].max())
    calibration_metrics = _selection_metrics(
        calibration_result,
        calibration.arrays["labels"],
        threshold,
        criteria.maximum_interval_width,
        support_min,
        support_max,
    )
    validation_grid = _grid(support, validation.arrays["sigma"])
    validation_result = _infer_checked(
        validation_grid, validation.arrays["observations"], "validation"
    )
    validation_metrics = _selection_metrics(
        validation_result,
        validation.arrays["labels"],
        threshold,
        criteria.maximum_interval_width,
        support_min,
        support_max,
    )
    qualification_failures = []
    if validation_metrics["acceptance"] < criteria.minimum_acceptance:
        qualification_failures.append("validation_acceptance_below_floor")
    accepted_coverage = validation_metrics["accepted_coverage"]
    if accepted_coverage is None or accepted_coverage < criteria.minimum_accepted_coverage:
        qualification_failures.append("validation_accepted_coverage_below_floor")

    observation_grid = _grid(support, observation.arrays["sigma"])
    observed = _infer_checked(
        observation_grid, observation.arrays["observation"][None, :], "observation"
    )
    lower = float(observed["lower"][0])
    upper = float(observed["upper"][0])
    score = float(observed["chi2"][0])
    try:
        with np.errstate(over="raise", invalid="raise"):
            width = upper - lower
    except FloatingPointError as error:
        raise ValueError("Observation interval width is numerically invalid") from error
    if not math.isfinite(width):
        raise ValueError("Observation interval width is non-finite")
    boundary = bool(lower <= support_min or upper >= support_max)
    reasons = list(qualification_failures)
    if score > threshold:
        reasons.append("observation_score_above_calibrated_threshold")
    if width > criteria.maximum_interval_width:
        reasons.append("observation_interval_above_width_cap")
    if boundary:
        reasons.append("observation_interval_touches_support_boundary")
    decision = "conditional_accept" if not reasons else "abstain"

    return {
        "schema_version": 1,
        "software": {"name": "tierra-plus", "version": __version__},
        "decision": decision,
        "reasons": reasons,
        "claim_boundary": (
            "Conditional finite-grid screen only; no generic OOD, model-family, "
            "continuous-parameter, or real-observation coverage guarantee."
        ),
        "metadata_identity_limit": (
            "Byte hashes are verified, but role independence relies on producer-declared "
            "dataset, source and physical-state identities."
        ),
        "criteria": {
            "score_quantile": criteria.score_quantile,
            "minimum_states": criteria.minimum_states,
            "minimum_acceptance": criteria.minimum_acceptance,
            "minimum_accepted_coverage": criteria.minimum_accepted_coverage,
            "maximum_interval_width": criteria.maximum_interval_width,
        },
        "score_calibration": {
            "threshold": threshold,
            "calibration": calibration_metrics,
            "validation": validation_metrics,
            "qualified": not qualification_failures,
            "qualification_failures": qualification_failures,
            "small_sample_note": (
                "Wilson intervals describe finite validation counts; gates use declared point "
                "estimates and do not create a shift-robust coverage guarantee."
            ),
        },
        "observation": {
            "mean": float(observed["mean"][0]),
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "conditional_envelope_lower": float(observed["envelope_lower"][0]),
            "conditional_envelope_upper": float(observed["envelope_upper"][0]),
            "minimum_chi2": score,
            "score_threshold": threshold,
            "score_pass": score <= threshold,
            "width_pass": width <= criteria.maximum_interval_width,
            "support_boundary": boundary,
        },
        "support": {
            "states": int(support.arrays["spectra"].shape[0]),
            "nuisance_states": (
                int(support.arrays["spectra"].shape[1])
                if support.arrays["spectra"].ndim == 3
                else 1
            ),
            "bins": int(support_wave.size),
            "label_min": support_min,
            "label_max": support_max,
            "likelihood": "independent Gaussian errors with a free additive depth offset",
        },
        "inputs": {
            bundle.role: {
                "dataset_id": bundle.manifest["dataset_id"],
                "source": bundle.manifest["source"],
                "source_fingerprint": bundle.manifest["source_fingerprint"],
                "data_sha256": bundle.archive_sha256,
                "manifest_sha256": bundle.manifest_sha256,
                "states": len(bundle.manifest["state_ids"]),
            }
            for bundle in bundles
        },
    }


def write_audit_report(report: dict[str, Any], output: Path) -> None:
    """Write a new report without replacing an existing artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(f"Refusing to replace existing output: {output}") from error
