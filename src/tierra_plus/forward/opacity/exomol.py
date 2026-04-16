"""ExoMol opacity metadata helpers."""

from __future__ import annotations

from pathlib import Path

from tierra_plus.forward.opacity import get_opacity_model
from tierra_plus.models import OpacityModelSpec


def pokazatel_h2o_spec(data_root: Path | None = None) -> OpacityModelSpec:
    """Return the tracked ExoMol POKAZATEL H2O manifest record."""

    return get_opacity_model("pokazatel_h2o", data_root=data_root)


def exoames_ch4_spec(data_root: Path | None = None) -> OpacityModelSpec:
    """Return the tracked ExoMol ExoAmes CH4 manifest record."""

    return get_opacity_model("exoames_ch4", data_root=data_root)
