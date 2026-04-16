"""Verify or fetch public Tierra+ data assets.

The first repository slice ships tracked fixtures and manifest records for larger public data
sources. Assets without a direct download URL must be acquired manually from their public provider.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from tierra_plus.forward.opacity import (
    default_data_root,
    load_opacity_manifest,
    verify_opacity_asset,
)


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
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(asset.download_url, path)
            verify_opacity_asset(asset, data_root=args.data_root)
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
