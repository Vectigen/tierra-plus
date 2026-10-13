from __future__ import annotations

import pytest

from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.backend_contract import (
    BackendContract,
    BackendSourceKind,
    backend_contract_for_petitradtrans,
    backend_contract_for_tierra,
)


def test_backend_contract_exposes_required_metadata() -> None:
    contract = BackendContract(
        backend_name="tierra",
        source_kind=BackendSourceKind.SUBMODULE,
        source_location="https://github.com/tierra-project/tierra",
        source_version="v0.0.0",
        installed=True,
        supported_modes=("transmission", "emission"),
    )

    payload = contract.to_dict()

    assert payload == {
        "backend_name": "tierra",
        "source_kind": "submodule",
        "source_location": "https://github.com/tierra-project/tierra",
        "source_version": "v0.0.0",
        "installed": True,
        "supported_modes": ["transmission", "emission"],
    }


def test_tierra_contract_reports_public_source_and_transmission_mode() -> None:
    contract = backend_contract_for_tierra()

    assert contract.source_location == "https://github.com/disruptiveplanets/tierra"
    assert contract.supported_modes == ("transmission",)


def test_petitradtrans_contract_reports_public_source_and_both_modes() -> None:
    contract = backend_contract_for_petitradtrans()

    assert contract.source_kind is BackendSourceKind.PIP
    assert contract.source_location == "https://gitlab.com/mauricemolli/petitRADTRANS"
    assert contract.supported_modes == ("transmission", "emission")


@pytest.mark.parametrize(
    ("source_location", "source_version"),
    [
        (None, "v0.0.0"),
        ("https://github.com/tierra-project/tierra", None),
        ("unknown", "unknown"),
    ],
)
def test_backend_contract_fails_when_source_identity_is_unknown(
    source_location: str | None,
    source_version: str | None,
) -> None:
    contract = BackendContract(
        backend_name="tierra",
        source_kind=BackendSourceKind.VENDORED,
        source_location=source_location,
        source_version=source_version,
        installed=False,
        supported_modes=("transmission",),
    )

    with pytest.raises(IntegrationNotConfiguredError, match="backend source.*unknown"):
        contract.ensure_source_is_known()
