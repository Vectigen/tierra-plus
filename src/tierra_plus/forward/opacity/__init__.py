"""Opacity manifest loading and verification utilities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from tierra_plus.models import OpacityModelSpec
from tierra_plus.provenance.hash import hash_path


def default_data_root() -> Path:
    """Return the repository's tracked public data root."""

    bundled = Path(__file__).resolve().parents[2] / "data" / "public"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[4] / "data" / "public"


def manifest_path(data_root: Path | None = None) -> Path:
    """Return the path to the tracked opacity manifest."""

    root = data_root if data_root is not None else default_data_root()
    return root / "opacity" / "manifest.json"


@lru_cache(maxsize=1)
def _load_cached_manifest() -> tuple[OpacityModelSpec, ...]:
    return _load_manifest_from_path(manifest_path())


def _load_manifest_from_path(path: Path) -> tuple[OpacityModelSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(OpacityModelSpec.from_dict(entry) for entry in payload["assets"])


def load_opacity_manifest(data_root: Path | None = None) -> tuple[OpacityModelSpec, ...]:
    """Load the public opacity manifest.

    Parameters
    ----------
    data_root
        Optional override for the public data root.
    """

    if data_root is None:
        return _load_cached_manifest()
    return _load_manifest_from_path(manifest_path(data_root))


def get_opacity_model(identifier: str, data_root: Path | None = None) -> OpacityModelSpec:
    """Return one opacity or benchmark asset from the public manifest."""

    for asset in load_opacity_manifest(data_root):
        if asset.identifier == identifier:
            return asset
    raise KeyError(f"Unknown opacity asset '{identifier}'.")


def verify_opacity_asset(asset: OpacityModelSpec, data_root: Path | None = None) -> Path:
    """Verify that a manifest asset exists and matches its recorded SHA-256 digest."""

    root = data_root if data_root is not None else default_data_root()
    path = asset.resolve_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Opacity asset '{asset.identifier}' is missing at '{path}'.")
    digest = hash_path(path)
    if digest != asset.sha256:
        raise ValueError(
            f"Opacity asset '{asset.identifier}' hash mismatch: "
            f"expected {asset.sha256}, got {digest}."
        )
    return path
