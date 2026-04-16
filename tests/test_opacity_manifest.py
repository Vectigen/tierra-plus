from __future__ import annotations

from pathlib import Path

import pytest

from tierra_plus.forward.opacity import (
    get_opacity_model,
    load_opacity_manifest,
    verify_opacity_asset,
)
from tierra_plus.models import OpacityModelSpec


def test_load_opacity_manifest_returns_expected_model() -> None:
    manifest = load_opacity_manifest()

    identifiers = {asset.identifier for asset in manifest}

    assert "hitran2020" in identifiers
    assert "synthetic-opacity-reference" in identifiers


def test_verify_opacity_asset_accepts_shipped_fixture() -> None:
    asset = get_opacity_model("synthetic-opacity-reference")

    verify_opacity_asset(asset)


def test_verify_opacity_asset_rejects_hash_mismatch(tmp_path: Path) -> None:
    asset = get_opacity_model("synthetic-opacity-reference")
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text('{"tampered": true}\n', encoding="utf-8")

    tampered_asset = OpacityModelSpec(
        identifier=asset.identifier,
        kind=asset.kind,
        provider=asset.provider,
        version=asset.version,
        species=asset.species,
        relative_path=tampered_path.relative_to(tmp_path),
        sha256=asset.sha256,
        source_url=asset.source_url,
        download_url=asset.download_url,
        manual_steps=asset.manual_steps,
        tracked_fixture=False,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_opacity_asset(tampered_asset, data_root=tmp_path)
