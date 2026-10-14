"""Prepare a bounded, provenance-locked Sonora Diamondback photometry grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from tierra_plus.provenance import hash_bytes, hash_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples/configs/sonora-diamondback-photometry-v1.json"
FloatArray = NDArray[np.float64]


def load_config(path: Path) -> dict[str, Any]:
    """Load the frozen protocol as a JSON object."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Sonora configuration must be a regular file, not a symbolic link")
    try:
        with path.open("rb") as handle:
            payload = handle.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("Sonora configuration exceeds the one-megabyte bound")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Sonora configuration is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Sonora configuration must be a schema-1 JSON object")
    source = value.get("source")
    if (
        value.get("status") != "frozen_before_inference"
        or not isinstance(source, dict)
        or source.get("license") != "CC-BY-4.0"
        or source.get("doi") != "10.5281/zenodo.12735103"
    ):
        raise ValueError("Sonora protocol status, DOI or license differs from the frozen contract")
    result = cast(dict[str, Any], value)
    result["_protocol_sha256"] = hash_bytes(payload)
    return result


def _download(url: str, maximum_bytes: int) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Sonora downloads require an unauthenticated HTTPS URL")
    request = urllib.request.Request(url, headers={"User-Agent": "tierra-plus/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            payload = bytes(response.read(maximum_bytes + 1))
    except OSError as error:
        raise ValueError("Sonora archive download failed") from error
    if len(payload) > maximum_bytes:
        raise ValueError("Sonora archive exceeds the configured download bound")
    return payload


def validate_record_metadata(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    """Match live Zenodo metadata to the frozen DOI, license and file identity."""
    source = cast(dict[str, Any], config["source"])
    archive = cast(dict[str, Any], config["archive"])
    record = metadata.get("metadata")
    files = metadata.get("files")
    if not isinstance(record, dict) or not isinstance(files, list):
        raise ValueError("Zenodo record metadata has an unexpected schema")
    license_record = record.get("license")
    if (
        record.get("doi") != source["doi"]
        or not isinstance(license_record, dict)
        or license_record.get("id") != "cc-by-4.0"
    ):
        raise ValueError("Zenodo DOI or license differs from the frozen protocol")
    matching = [
        item for item in files if isinstance(item, dict) and item.get("key") == archive["file"]
    ]
    if len(matching) != 1:
        raise ValueError("Zenodo record does not contain exactly one photometry archive")
    item = matching[0]
    if item.get("size") != archive["bytes"] or item.get("checksum") != f"md5:{archive['md5']}":
        raise ValueError("Zenodo photometry identity differs from the frozen protocol")


def verify_record_metadata(config: dict[str, Any]) -> None:
    """Fetch and validate the bounded Zenodo record metadata."""
    source = cast(dict[str, Any], config["source"])
    payload = _download(str(source["record_api"]), 1_000_000)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Zenodo record metadata is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Zenodo record metadata must be a JSON object")
    validate_record_metadata(cast(dict[str, Any], value), config)


def read_archive(path: Path | None, config: dict[str, Any]) -> bytes:
    """Read one local archive or fetch it, with the same strict byte checks."""
    archive = cast(dict[str, Any], config["archive"])
    maximum = int(archive["maximum_download_bytes"])
    if path is None:
        payload = _download(str(archive["url"]), maximum)
    else:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Sonora archive must be a regular file, not a symbolic link")
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
        if len(payload) > maximum:
            raise ValueError("Sonora archive exceeds the configured download bound")
    if len(payload) != int(archive["bytes"]):
        raise ValueError("Sonora archive byte count does not match the frozen record")
    if hashlib.md5(payload, usedforsecurity=False).hexdigest() != archive["md5"]:
        raise ValueError("Sonora archive MD5 does not match the Zenodo record")
    if hash_bytes(payload) != archive["sha256"]:
        raise ValueError("Sonora archive SHA-256 does not match the frozen protocol")
    return payload


def _member_payloads(payload: bytes, config: dict[str, Any]) -> dict[str, bytes]:
    expected = {str(item["path"]): item for item in cast(list[dict[str, Any]], config["members"])}
    archive_config = cast(dict[str, Any], config["archive"])
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > int(archive_config["maximum_members"]):
                raise ValueError("Sonora archive has too many members")
            if len(names) != len(set(names)):
                raise ValueError("Sonora archive contains duplicate member names")
            if not expected.keys() <= set(names):
                raise ValueError("Sonora archive is missing a frozen photometry member")
            if sum(member.file_size for member in members) > int(
                archive_config["maximum_uncompressed_bytes"]
            ):
                raise ValueError("Sonora archive exceeds the uncompressed size bound")
            if any(
                member.flag_bits & 0x1
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or member.file_size > 200_000
                for member in members
            ):
                raise ValueError("Sonora archive member violates safety bounds")
            result = {name: archive.read(name) for name in expected}
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError("Sonora input is not a valid bounded ZIP archive") from error
    for name, value in result.items():
        record = expected[name]
        if len(value) != int(record["bytes"]) or hash_bytes(value) != record["sha256"]:
            raise ValueError(f"Sonora member does not match its frozen identity: {name}")
    return result


def _parse_member(payload: bytes, record: dict[str, Any], columns: list[str]) -> FloatArray:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError("Sonora photometry member is not UTF-8 text") from error
    if len(lines) < 5 or lines[3].split() != columns:
        raise ValueError("Sonora photometry columns differ from the frozen schema")
    try:
        values = np.loadtxt(StringIO("\n".join(lines[4:])), dtype=np.float64)
    except ValueError as error:
        raise ValueError("Sonora photometry rows are not a rectangular numeric table") from error
    if values.shape != (480, len(columns)) or not np.isfinite(values).all():
        raise ValueError("Sonora photometry member has unexpected shape or non-finite values")
    if not np.all(values[:, 3] == float(record["metallicity_dex"])):
        raise ValueError("Sonora member metallicity disagrees with its frozen identity")
    return values


def ingest(payload: bytes, config: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """Parse the official archive and apply only predeclared coordinate rules."""
    columns = [str(value) for value in cast(list[str], config["columns"])]
    records = cast(list[dict[str, Any]], config["members"])
    members = _member_payloads(payload, config)
    values = np.concatenate(
        [_parse_member(members[str(record["path"])], record, columns) for record in records]
    )
    grid = cast(dict[str, Any], config["expected_grid"])
    expected_axes = {
        0: np.asarray(grid["temperatures_k"], dtype=np.float64),
        1: np.asarray(grid["gravity_m_s2"], dtype=np.float64),
        2: np.asarray(grid["fsed"], dtype=np.float64),
        3: np.asarray(grid["metallicity_dex"], dtype=np.float64),
    }
    if values.shape[0] != int(grid["rows"]):
        raise ValueError("Sonora combined grid row count differs from the frozen schema")
    for column, expected in expected_axes.items():
        if not np.array_equal(np.unique(values[:, column]), expected):
            raise ValueError("Sonora coordinate axis differs from the frozen schema")
    coordinates = values[:, :4]
    if len(np.unique(coordinates, axis=0)) != len(values):
        raise ValueError("Sonora grid contains duplicate physical coordinates")

    feature_names = np.asarray(config["photometric_features"], dtype="U16")
    feature_start = columns.index(str(feature_names[0]))
    magnitudes = values[:, feature_start:]
    if magnitudes.shape[1] != len(feature_names):
        raise ValueError("Sonora photometric feature count differs from the frozen schema")
    teff = values[:, 0]
    gravity = values[:, 1]
    fsed = values[:, 2].astype(np.int16)
    metallicity = values[:, 3]
    logg = np.log10(gravity) + 2.0
    state_ids = np.asarray(
        [
            f"teff={temperature:.0f}|grav={g:.1f}|feh={feh:+.1f}|fsed={cloud:d}"
            for temperature, g, feh, cloud in zip(teff, gravity, metallicity, fsed, strict=True)
        ],
        dtype="U48",
    )
    state_sha256 = np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in values], dtype="U64"
    )
    partition = cast(dict[str, dict[str, Any]], config["partition"])

    def indices(role: str) -> NDArray[np.int64]:
        rule = partition[role]
        cloud_ok = np.isin(fsed, np.asarray(rule["fsed"], dtype=np.int16))
        gravity_ok = np.isin(gravity, np.asarray(rule["gravity_m_s2"], dtype=np.float64))
        result = np.flatnonzero(cloud_ok & gravity_ok).astype(np.int64)
        if len(result) != int(rule["expected_states"]):
            raise ValueError(f"Sonora {role} partition count differs from its frozen rule")
        return result

    support = indices("support")
    calibration = indices("calibration")
    validation = indices("validation")
    challenge = indices("cloud_challenge")
    if (
        set(support) & set(calibration)
        or set(support) & set(validation)
        or set(calibration) & set(validation)
    ):
        raise ValueError("Sonora cloud-free partitions overlap")
    paired_validation = {(teff[index], gravity[index], metallicity[index]) for index in validation}
    paired_challenge = {(teff[index], gravity[index], metallicity[index]) for index in challenge}
    if paired_validation != paired_challenge:
        raise ValueError("Sonora cloud challenge is not paired to every validation coordinate")
    if len(state_ids) != len(set(state_ids.tolist())) or len(state_sha256) != len(
        set(state_sha256.tolist())
    ):
        raise ValueError("Sonora state identities are not unique")
    return {
        "feature_names": feature_names,
        "magnitudes": magnitudes,
        "effective_temperature_k": teff,
        "gravity_m_s2": gravity,
        "logg_cgs": logg,
        "metallicity_dex": metallicity,
        "fsed": fsed,
        "age_gyr": values[:, 5],
        "mass_jupiter": values[:, 6],
        "radius_jupiter": values[:, 7],
        "log_luminosity_solar": values[:, 8],
        "state_ids": state_ids,
        "state_sha256": state_sha256,
        "support_indices": support,
        "calibration_indices": calibration,
        "validation_indices": validation,
        "cloud_challenge_indices": challenge,
    }


def _stable_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    with zipfile.ZipFile(path, "x") as archive:
        for name in sorted(arrays):
            handle = BytesIO()
            np.save(handle, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, handle.getvalue())


def write_dataset(
    output: Path,
    arrays: dict[str, NDArray[Any]],
    payload: bytes,
    config: dict[str, Any],
) -> None:
    """Write a new canonical dataset and provenance manifest without replacement."""
    output.mkdir(parents=True, exist_ok=False)
    try:
        data_path = output / "grid.npz"
        _stable_npz(data_path, arrays)
        partition = cast(dict[str, dict[str, Any]], config["partition"])
        manifest = {
            "schema_version": 1,
            "status": "prepared_without_inference",
            "source": config["source"],
            "source_archive_sha256": hash_bytes(payload),
            "source_archive_bytes": len(payload),
            "protocol_sha256": config["_protocol_sha256"],
            "grid_sha256": hash_path(data_path),
            "states": len(arrays["effective_temperature_k"]),
            "features": arrays["feature_names"].tolist(),
            "partition_counts": {
                role: int(partition[role]["expected_states"])
                for role in ("support", "calibration", "validation", "cloud_challenge")
            },
            "prospective_rule": config["prospective_rule"],
            "claim_boundary": config["claim_boundary"],
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Verified local photometry.zip; fetch if omitted"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-record",
        action="store_true",
        help="also confirm the live Zenodo DOI, license and archive identity",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.verify_record or args.archive is None:
        verify_record_metadata(config)
    payload = read_archive(args.archive, config)
    arrays = ingest(payload, config)
    write_dataset(args.output, arrays, payload, config)
    print(
        json.dumps(
            {
                "status": "prepared_without_inference",
                "output": str(args.output),
                "states": len(arrays["effective_temperature_k"]),
                "cloud_challenge_states": len(arrays["cloud_challenge_indices"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
