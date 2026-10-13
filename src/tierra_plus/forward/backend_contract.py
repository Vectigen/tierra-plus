"""Backend integration contract metadata for Tierra+."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tierra_plus.exceptions import IntegrationNotConfiguredError


class BackendSourceKind(StrEnum):
    """Kinds of backend sources supported by Tierra+."""

    SUBMODULE = "submodule"
    VENDORED = "vendored"
    PIP = "pip"
    GIT = "git"


@dataclass(frozen=True)
class BackendContract:
    """Describe the integration contract for a concrete backend.

    Parameters
    ----------
    backend_name
        Human-readable backend name used in errors and docs.
    source_kind
        Where the backend is expected to come from.
    source_location
        Canonical source location or repository URL when known.
    source_version
        Pinned version, tag, commit hash, or release identifier when known.
    installed
        Whether the backend is currently importable in the active environment.
    supported_modes
        Call modes that the backend is expected to implement.
    """

    backend_name: str
    source_kind: BackendSourceKind
    source_location: str | None
    source_version: str | None
    installed: bool
    supported_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_modes",
            tuple(str(mode) for mode in self.supported_modes),
        )
        if not self.backend_name.strip():
            raise ValueError("backend_name must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract into JSON-compatible primitives."""

        return {
            "backend_name": self.backend_name,
            "source_kind": self.source_kind.value,
            "source_location": self.source_location,
            "source_version": self.source_version,
            "installed": self.installed,
            "supported_modes": list(self.supported_modes),
        }

    def ensure_source_is_known(self) -> None:
        """Raise if the backend source location or version has not been resolved."""

        unknown_values = {None, "", "unknown"}
        if self.source_location in unknown_values or self.source_version in unknown_values:
            raise IntegrationNotConfiguredError(
                f"{self.backend_name} backend source is unknown; "
                "resolve the source location and version before enabling this backend."
            )


def backend_contract_for_tierra() -> BackendContract:
    """Return the current Tierra backend contract placeholder."""

    return BackendContract(
        backend_name="tierra",
        source_kind=BackendSourceKind.GIT,
        source_location="https://github.com/disruptiveplanets/tierra",
        source_version="unknown",
        installed=False,
        supported_modes=("transmission",),
    )


def backend_contract_for_petitradtrans() -> BackendContract:
    """Return the current petitRADTRANS backend contract placeholder."""

    return BackendContract(
        backend_name="petitRADTRANS",
        source_kind=BackendSourceKind.PIP,
        source_location="https://gitlab.com/mauricemolli/petitRADTRANS",
        source_version="unknown",
        installed=False,
        supported_modes=("transmission", "emission"),
    )
