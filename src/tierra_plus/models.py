"""Shared typed models for Tierra+."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coerce_float_tuple(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class Spectrum:
    """One-dimensional spectrum container.

    Parameters
    ----------
    wavelength_microns
        Sample wavelengths in microns.
    flux
        Spectral flux or transit depth values sampled on `wavelength_microns`.
    uncertainty
        Optional 1-sigma uncertainties aligned with `flux`.

    Examples
    --------
    >>> spectrum = Spectrum([1.0, 1.2], [0.9, 0.8], [0.01, 0.01])
    >>> spectrum.flux[0]
    0.9
    """

    wavelength_microns: tuple[float, ...] | list[float]
    flux: tuple[float, ...] | list[float]
    uncertainty: tuple[float, ...] | list[float] | None = None

    def __post_init__(self) -> None:
        wavelengths = _coerce_float_tuple(list(self.wavelength_microns))
        flux = _coerce_float_tuple(list(self.flux))
        uncertainty = (
            _coerce_float_tuple(list(self.uncertainty)) if self.uncertainty is not None else None
        )
        if len(wavelengths) == 0:
            raise ValueError("Spectrum must contain at least one wavelength sample.")
        if len(wavelengths) != len(flux):
            raise ValueError("Spectrum wavelength and flux arrays must have the same length.")
        if uncertainty is not None and len(uncertainty) != len(wavelengths):
            raise ValueError("Spectrum uncertainty array must have the same length as flux.")
        if uncertainty is not None and any(value <= 0.0 for value in uncertainty):
            raise ValueError("Spectrum uncertainties must be strictly positive.")
        object.__setattr__(self, "wavelength_microns", wavelengths)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spectrum into JSON-compatible primitives."""

        return {
            "wavelength_microns": list(self.wavelength_microns),
            "flux": list(self.flux),
            "uncertainty": list(self.uncertainty) if self.uncertainty is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Spectrum:
        """Construct a spectrum from a mapping payload."""

        return cls(
            wavelength_microns=list(payload["wavelength_microns"]),
            flux=list(payload["flux"]),
            uncertainty=list(payload["uncertainty"]) if payload.get("uncertainty") else None,
        )


@dataclass(frozen=True)
class AtmosphericState:
    """Compact atmospheric state used by the initial forward-model interface.

    Parameters
    ----------
    temperature_kelvin
        Atmospheric temperature in kelvin.
    radius_rjup
        Planet radius in Jupiter radii.
    gravity_m_s2
        Surface gravity in meters per second squared.
    composition
        Species abundance map keyed by molecule name.
    cloud_top_pressure_bar
        Optional cloud-top pressure in bar.

    Examples
    --------
    >>> state = AtmosphericState(500.0, 0.2, 9.8, {"H2O": 0.5, "N2": 0.5})
    >>> round(sum(state.composition.values()), 1)
    1.0
    """

    temperature_kelvin: float
    radius_rjup: float
    gravity_m_s2: float
    composition: Mapping[str, float]
    cloud_top_pressure_bar: float | None = None

    def __post_init__(self) -> None:
        if self.temperature_kelvin <= 0.0:
            raise ValueError("AtmosphericState.temperature_kelvin must be positive.")
        if self.radius_rjup <= 0.0:
            raise ValueError("AtmosphericState.radius_rjup must be positive.")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError("AtmosphericState.gravity_m_s2 must be positive.")
        if not self.composition:
            raise ValueError("AtmosphericState.composition must not be empty.")
        normalized = {str(key): float(value) for key, value in self.composition.items()}
        if any(value < 0.0 for value in normalized.values()):
            raise ValueError("AtmosphericState composition fractions must be non-negative.")
        object.__setattr__(self, "composition", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state into JSON-compatible primitives."""

        return {
            "temperature_kelvin": self.temperature_kelvin,
            "radius_rjup": self.radius_rjup,
            "gravity_m_s2": self.gravity_m_s2,
            "composition": dict(self.composition),
            "cloud_top_pressure_bar": self.cloud_top_pressure_bar,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AtmosphericState:
        """Construct a state from a mapping payload."""

        return cls(
            temperature_kelvin=float(payload["temperature_kelvin"]),
            radius_rjup=float(payload["radius_rjup"]),
            gravity_m_s2=float(payload["gravity_m_s2"]),
            composition=dict(payload["composition"]),
            cloud_top_pressure_bar=(
                float(payload["cloud_top_pressure_bar"])
                if payload.get("cloud_top_pressure_bar") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for a forward-model backend integration."""

    backend_name: str
    module_name: str | None = None
    data_root: Path | None = None
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpacityModelSpec:
    """Opacity or benchmark asset metadata loaded from the public manifest."""

    identifier: str
    kind: str
    provider: str
    version: str
    species: tuple[str, ...]
    relative_path: Path
    sha256: str
    source_url: str
    download_url: str | None
    manual_steps: str | None
    tracked_fixture: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpacityModelSpec:
        """Construct a manifest record from JSON-compatible primitives."""

        return cls(
            identifier=str(payload["identifier"]),
            kind=str(payload["kind"]),
            provider=str(payload["provider"]),
            version=str(payload["version"]),
            species=tuple(str(entry) for entry in payload["species"]),
            relative_path=Path(str(payload["relative_path"])),
            sha256=str(payload["sha256"]),
            source_url=str(payload["source_url"]),
            download_url=(
                str(payload["download_url"]) if payload.get("download_url") is not None else None
            ),
            manual_steps=(
                str(payload["manual_steps"]) if payload.get("manual_steps") is not None else None
            ),
            tracked_fixture=bool(payload["tracked_fixture"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest record into JSON-compatible primitives."""

        return {
            "identifier": self.identifier,
            "kind": self.kind,
            "provider": self.provider,
            "version": self.version,
            "species": list(self.species),
            "relative_path": self.relative_path.as_posix(),
            "sha256": self.sha256,
            "source_url": self.source_url,
            "download_url": self.download_url,
            "manual_steps": self.manual_steps,
            "tracked_fixture": self.tracked_fixture,
        }

    def resolve_path(self, data_root: Path) -> Path:
        """Resolve the asset path beneath the configured public data root."""

        return data_root / self.relative_path
