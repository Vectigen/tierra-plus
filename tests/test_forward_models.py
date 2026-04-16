from __future__ import annotations

import pytest

from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.prt_wrapper import PetitRadTransForwardModel
from tierra_plus.forward.tierra_wrapper import TierraForwardModel
from tierra_plus.models import AtmosphericState, BackendConfig


def build_state() -> AtmosphericState:
    return AtmosphericState(
        temperature_kelvin=600.0,
        radius_rjup=0.3,
        gravity_m_s2=8.5,
        composition={"H2": 0.9, "H2O": 0.1},
    )


def test_tierra_wrapper_fails_loudly_when_backend_is_missing() -> None:
    model = TierraForwardModel(
        BackendConfig(backend_name="tierra", module_name="module_that_does_not_exist")
    )

    with pytest.raises(IntegrationNotConfiguredError, match="tierra"):
        model.transmission_spectrum(build_state())


def test_prt_wrapper_fails_loudly_when_backend_is_missing() -> None:
    model = PetitRadTransForwardModel(
        BackendConfig(backend_name="petitRADTRANS", module_name="module_that_does_not_exist")
    )

    with pytest.raises(IntegrationNotConfiguredError, match="petitRADTRANS"):
        model.emission_spectrum(build_state())
