"""Prepare bounded, checksum-verified public PICASO grids without extracting ZIP paths.

Run with ``uv run --with h5py==3.16.0 python -m examples.prepare_physical_grid``.
Supply the separately downloaded author archive; this script never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

from tierra_plus.diagnostics.physical_grid import shifted_bin_integrals

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_HASH = "02fa509fc8cd81eaa8451a378a72cb3df7488cfcf5ea83c3b74c978f978d884c"
ARCHIVES = {
    "rcte": ("RCTE_cloud_free.zip", 31104898, "4b34c96d22575e6ea65b1f8d3c058092"),
    "photochemical": ("photochem_cloud_free.zip", 18730827, "74ec83af55eeca6127629f16f01c1409"),
}


def prepare(archive: Path, output: Path, family: str) -> dict:
    """Only spectra/geometry are prepared; no simulated observations or fit outcomes."""
    name, size, md5 = ARCHIVES[family]
    if archive.stat().st_size != size:
        raise ValueError(f"Expected {name}: {size} bytes")
    payload = archive.read_bytes()
    if hashlib.md5(payload).hexdigest() != md5:
        raise ValueError("Author archive checksum mismatch")
    observation = ROOT / "examples/data/wasp39-prism/FIREFLy_transit_spec.h5"
    if hashlib.sha256(observation.read_bytes()).hexdigest() != OBSERVATION_HASH:
        raise ValueError("Published PRISM input checksum mismatch")
    with h5py.File(observation) as handle:
        wave = np.asarray(handle["wavelength"])
        errors = np.asarray(handle["transit_depth_uncertainty"])
        # Deliberately never read observed transit_depth.
    edges = np.concatenate(
        (
            [wave[0] - (wave[1] - wave[0]) / 2],
            (wave[:-1] + wave[1:]) / 2,
            [wave[-1] + (wave[-1] - wave[-2]) / 2],
        )
    )
    keep = edges[:-1] >= 2.3
    first = np.flatnonzero(keep)[0]
    edges, errors = edges[first:], errors[keep]
    shift_bound = float(np.median(np.diff(edges)) / 2)
    shifts = np.array([-1, 0, 1, 0.5]) * shift_bound
    spectra, states = [], []
    with zipfile.ZipFile(io.BytesIO(payload)) as handle:
        members = handle.infolist()
        names = [item.filename for item in members]
        if len(members) > 1024 or len(names) != len(set(names)):
            raise ValueError("Archive has duplicate or excessive members")
        if sum(item.file_size for item in members) > 100_000_000:
            raise ValueError("Archive expansion exceeds the 100 MB budget")
        selected = sorted(
            (
                x
                for x in members
                if x.filename.endswith(".nc") and not x.filename.startswith("__MACOSX/")
            ),
            key=lambda x: x.filename,
        )
        if not selected or len(selected) > 384:
            raise ValueError("Unexpected physical-state count")
        for item in selected:
            if item.file_size > 1_000_000 or item.is_dir():
                raise ValueError("Unexpected member size/type")
            raw = handle.read(item)
            with h5py.File(io.BytesIO(raw)) as model:
                model_wave = np.asarray(model["wavelength"])
                model_depth = np.asarray(model["transit_depth"])
                params = json.loads(model.attrs["planet_params"])
                code = json.loads(model.attrs["code"])
            match = re.search(r"_mh_([+-][0-9.]+)_CO_([0-9.]+)_", item.filename)
            if match is None or code["spectrum"] != "PICASO":
                raise ValueError("Unexpected model naming/provenance")
            log_metallicity = float(match[1])
            if not np.isclose(np.log10(params["mh"]), log_metallicity, atol=0.002):
                raise ValueError("Filename metallicity conflicts with physical metadata")
            order = np.argsort(model_wave)
            binned = shifted_bin_integrals(model_wave[order], model_depth[order], edges, shifts)
            spectra.append(binned)
            states.append(
                {
                    "id": item.filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "log_metallicity": log_metallicity,
                    "carbon_to_oxygen": params["cto"],
                    "tint": params["tint"],
                    "heat_redistribution": params["heat_redis"],
                    "logkzz": params.get("logkzz"),
                    "code": code,
                }
            )
    if family == "rcte":
        split = {
            "train": [i for i, x in enumerate(states) if x["tint"] in (100, 300)],
            "validation": [
                i
                for i, x in enumerate(states)
                if x["tint"] == 200 and x["heat_redistribution"] == 0.4
            ],
            "test": [
                i
                for i, x in enumerate(states)
                if x["tint"] == 200 and x["heat_redistribution"] == 0.5
            ],
        }
        if [len(x) for x in split.values()] != [128, 32, 32]:
            raise ValueError("Physical grid does not match the frozen 128/32/32 split")
        if sorted(sum(split.values(), [])) != list(range(len(states))):
            raise ValueError("Physical split does not partition the grid")
    else:
        split = {"external": list(range(len(states)))}
    manifest = {
        "family": family,
        "source_doi": "10.5281/zenodo.7236759",
        "license": "CC-BY-4.0",
        "authors": "Mukherjee, Moran, Ohno, Batalha, Fortney (2022)",
        "archive": name,
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "prism_sha256": OBSERVATION_HASH,
        "states": states,
        "splits": split,
        "shift_bound_um": shift_bound,
        "shifts_um": shifts.tolist(),
        "response": "midpoint top-hat bins, lower edge >= 2.3 um; no LSF/covariance",
        "quantity": "dimensionless transit depth; metallicity label log10(relative to solar)",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "grid.npz", spectra=np.asarray(spectra), sigma=errors, edges=edges)
    manifest["grid_sha256"] = hashlib.sha256((output / "grid.npz").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=ARCHIVES, default="rcte")
    args = parser.parse_args()
    result = prepare(args.archive, args.output, args.family)
    print(
        json.dumps(
            {
                "family": result["family"],
                "states": len(result["states"]),
                "splits": {k: len(v) for k, v in result["splits"].items()},
            }
        )
    )
