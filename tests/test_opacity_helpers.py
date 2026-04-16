from __future__ import annotations

from pathlib import Path

from tierra_plus.forward.opacity.exomol import exoames_ch4_spec, pokazatel_h2o_spec
from tierra_plus.forward.opacity.hitemp import hitemp_spec
from tierra_plus.forward.opacity.hitran import hitran2020_spec


def test_opacity_helper_modules_resolve_manifest_entries() -> None:
    assert hitran2020_spec().provider == "HITRAN"
    assert hitemp_spec().provider == "HITEMP"
    assert pokazatel_h2o_spec().species == ("H2O",)
    assert exoames_ch4_spec().relative_path == Path("opacity/catalogs/exoames_ch4_catalog.txt")
