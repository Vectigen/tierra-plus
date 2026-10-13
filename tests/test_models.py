from __future__ import annotations

import pytest

from tierra_plus.models import AtmosphericState, BackendConfig, Spectrum


def test_spectrum_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        Spectrum(
            wavelength_microns=[1.0, 2.0],
            flux=[0.1],
            uncertainty=[0.01, 0.01],
        )


def test_atmospheric_state_round_trip_serialization() -> None:
    state = AtmosphericState(
        temperature_kelvin=500.0,
        radius_rjup=0.25,
        gravity_m_s2=9.8,
        composition={"H2O": 0.2, "N2": 0.8},
        cloud_top_pressure_bar=0.1,
    )

    payload = state.to_dict()

    restored = AtmosphericState.from_dict(payload)

    assert restored == state


def test_backend_config_round_trip_serialization() -> None:
    config = BackendConfig(
        backend_name="petitRADTRANS",
        module_name="petitRADTRANS",
        source_kind="pip",
        source_location="https://gitlab.com/mauricemolli/petitRADTRANS",
        source_revision="3.3.3",
        install_status="partially_configured",
        extra={"profile": "retrieval"},
    )

    payload = config.to_dict()

    restored = BackendConfig.from_dict(payload)

    assert restored == config


def test_backend_config_rejects_unknown_install_status() -> None:
    with pytest.raises(ValueError, match="install_status"):
        BackendConfig(backend_name="tierra", install_status="broken")
