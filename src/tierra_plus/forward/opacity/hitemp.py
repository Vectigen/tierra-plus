"""HITEMP opacity metadata helpers."""

from __future__ import annotations

from pathlib import Path

from tierra_plus.forward.opacity import get_opacity_model
from tierra_plus.models import OpacityModelSpec


def hitemp_spec(data_root: Path | None = None) -> OpacityModelSpec:
    """Return the tracked HITEMP manifest record."""

    return get_opacity_model("hitemp", data_root=data_root)
