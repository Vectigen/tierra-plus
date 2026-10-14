"""Verify or fetch public Tierra+ data assets.

The first repository slice ships tracked fixtures and manifest records for larger public data
sources. Assets without a direct download URL must be acquired manually from their public provider.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from tierra_plus.forward.opacity import (
    default_data_root,
    load_opacity_manifest,
    verify_opacity_asset,
)
from tierra_plus.models import OpacityModelSpec
from tierra_plus.provenance.hash import hash_path


def download_verified_asset(asset: OpacityModelSpec, data_root: Path) -> Path:
    """Publish a downloaded asset only after verification, without overwriting files."""
    if asset.download_url is None:
        raise ValueError("A direct download URL is required")
    parsed = urlsplit(asset.download_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Public asset downloads require an unauthenticated HTTPS URL")
    path = asset.resolve_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".tierra-download-", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        urllib.request.urlretrieve(asset.download_url, temporary)  # nosec B310
        if hash_path(temporary) != asset.sha256:
            raise ValueError(f"Opacity asset '{asset.identifier}' download hash mismatch")
        # An existing destination, including a concurrently created symlink, must
        # not be replaced or followed. Both names are on the same filesystem.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the public-data bootstrap helper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    return parser


def main() -> int:
    """Run the bootstrap workflow for tracked public assets."""

    args = build_parser().parse_args()
    failures: list[str] = []
    for asset in load_opacity_manifest(args.data_root):
        path = asset.resolve_path(args.data_root)
        if path.exists():
            verify_opacity_asset(asset, data_root=args.data_root)
            print(f"verified {asset.identifier}: {path}")
            continue
        if args.verify_only:
            failures.append(f"missing {asset.identifier}: {path}")
            continue
        if asset.download_url is not None:
            download_verified_asset(asset, args.data_root)
            print(f"downloaded {asset.identifier}: {path}")
            continue
        failures.append(
            "manual action required for "
            f"{asset.identifier}: {asset.manual_steps or 'no instructions recorded'}"
        )
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
