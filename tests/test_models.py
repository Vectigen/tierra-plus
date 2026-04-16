from __future__ import annotations

import pytest

from tierra_plus.models import AtmosphericState, Spectrum


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
