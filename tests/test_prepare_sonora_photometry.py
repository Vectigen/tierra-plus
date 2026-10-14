from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tierra_plus.provenance import hash_bytes, hash_path


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_sonora_photometry.py"
    spec = importlib.util.spec_from_file_location("prepare_sonora_photometry", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Sonora preparation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table(config: dict[str, Any], metallicity: float) -> bytes:
    columns = config["columns"]
    rows: list[str] = []
    for temperature in config["expected_grid"]["temperatures_k"]:
        for gravity in config["expected_grid"]["gravity_m_s2"]:
            for fsed in config["expected_grid"]["fsed"]:
                leading = [
                    temperature,
                    gravity,
                    fsed,
                    metallicity,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -5.0,
                ]
                magnitudes = [
                    temperature / 100 + np.log10(gravity) + metallicity + fsed / 100 + i / 1000
                    for i in range(16)
                ]
                rows.append("\t".join(str(value) for value in [*leading, *magnitudes]))
    return (
        "Synthetic Sonora-shaped fixture\n"
        "Evolution file: fixture\n"
        "Photometric filters\n" + "\t".join(columns) + "\n" + "\n".join(rows) + "\n"
    ).encode()


def _fixture(tmp_path: Path) -> tuple[ModuleType, dict[str, Any], Path, Path]:
    module = _load_module()
    base = Path(__file__).resolve().parents[1] / "examples/configs"
    config = json.loads((base / "sonora-diamondback-photometry-v1.json").read_text())
    archive_path = tmp_path / "photometry.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in config["members"]:
            payload = _table(config, float(record["metallicity_dex"]))
            archive.writestr(record["path"], payload)
            record["bytes"] = len(payload)
            record["sha256"] = hash_bytes(payload)
    payload = archive_path.read_bytes()
    config["archive"]["bytes"] = len(payload)
    config["archive"]["md5"] = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    config["archive"]["sha256"] = hash_bytes(payload)
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return module, module.load_config(config_path), config_path, archive_path


def test_prepares_frozen_partitions_without_running_inference(tmp_path: Path) -> None:
    module, config, _, archive_path = _fixture(tmp_path)
    payload = module.read_archive(archive_path, config)
    arrays = module.ingest(payload, config)

    assert arrays["magnitudes"].shape == (1440, 16)
    assert len(arrays["support_indices"]) == 144
    assert len(arrays["calibration_indices"]) == 48
    assert len(arrays["validation_indices"]) == 48
    assert len(arrays["cloud_challenge_indices"]) == 240
    assert np.all(arrays["fsed"][arrays["validation_indices"]] == 0)
    assert set(arrays["fsed"][arrays["cloud_challenge_indices"]]) == {1, 2, 3, 4, 8}

    first = tmp_path / "first"
    second = tmp_path / "second"
    module.write_dataset(first, arrays, payload, config)
    module.write_dataset(second, arrays, payload, config)
    assert hash_path(first / "grid.npz") == hash_path(second / "grid.npz")
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "prepared_without_inference"
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert manifest["partition_counts"] == {
        "support": 144,
        "calibration": 48,
        "validation": 48,
        "cloud_challenge": 240,
    }
    with pytest.raises(FileExistsError):
        module.write_dataset(first, arrays, payload, config)


def test_rejects_archive_identity_change_before_parsing(tmp_path: Path) -> None:
    module, config, _, archive_path = _fixture(tmp_path)
    changed = tmp_path / "changed.zip"
    payload = bytearray(archive_path.read_bytes())
    payload[-1] ^= 1
    changed.write_bytes(payload)

    with pytest.raises(ValueError, match="does not match"):
        module.read_archive(changed, config)


def test_rejects_duplicate_zip_members_before_reading_tables(tmp_path: Path) -> None:
    module, config, _, archive_path = _fixture(tmp_path)
    duplicate = config["members"][0]["path"]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr(duplicate, b"duplicate")
    payload = archive_path.read_bytes()
    config["archive"]["bytes"] = len(payload)
    config["archive"]["md5"] = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    config["archive"]["sha256"] = hash_bytes(payload)

    with pytest.raises(ValueError, match="duplicate member"):
        module.ingest(payload, config)


def test_validates_live_record_doi_license_and_file_identity(tmp_path: Path) -> None:
    module, config, _, _ = _fixture(tmp_path)
    metadata = {
        "metadata": {"doi": config["source"]["doi"], "license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": config["archive"]["file"],
                "size": config["archive"]["bytes"],
                "checksum": f"md5:{config['archive']['md5']}",
            }
        ],
    }
    module.validate_record_metadata(metadata, config)

    changed = copy.deepcopy(metadata)
    changed["metadata"]["license"]["id"] = "unknown"
    with pytest.raises(ValueError, match="DOI or license"):
        module.validate_record_metadata(changed, config)


def test_sonora_network_fetch_rejects_non_https_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    opened = False

    def open_url(*args: object, **kwargs: object) -> None:
        nonlocal opened
        opened = True

    monkeypatch.setattr(module.urllib.request, "urlopen", open_url)
    with pytest.raises(ValueError, match="HTTPS"):
        module._download("http://example.invalid/archive.zip", 1000)
    assert opened is False


def test_public_preparation_record_binds_protocol_and_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "examples/results/sonora-photometry-preparation/record.json").read_text()
    )

    assert record["inference_run"] is False
    assert record["status"] == "prepared_without_inference"
    assert record["protocol_sha256"] == hash_path(
        root / "examples/configs/sonora-diamondback-photometry-v1.json"
    )
    assert record["adapter_sha256"] == hash_path(root / "scripts/prepare_sonora_photometry.py")
