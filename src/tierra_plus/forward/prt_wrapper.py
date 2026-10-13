"""Adapter seam for the `petitRADTRANS` backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.base import ForwardModel
from tierra_plus.models import AtmosphericState, Spectrum


class PetitRadTransForwardModel(ForwardModel):
    """Wrap `petitRADTRANS` behind the Tierra+ forward-model interface.

    Notes
    -----
    The API seam validates environment setup and raises `IntegrationNotConfiguredError` until the
    backend wiring is implemented and tested against the selected public inputs.
    """

    def _validate_backend(self) -> None:
        module_name = self.config.module_name or "petitRADTRANS"
        if importlib.util.find_spec(module_name) is None:
            source_hint = (
                f" Expected source: {self.config.source_location}."
                if self.config.source_location is not None
                else ""
            )
            raise IntegrationNotConfiguredError(
                f"petitRADTRANS backend module '{module_name}' is not installed. "
                "Install and validate the backend before use."
                f"{source_hint}"
            )
        if self.config.data_root is not None and not Path(self.config.data_root).exists():
            raise IntegrationNotConfiguredError(
                f"Configured petitRADTRANS data root '{self.config.data_root}' does not exist."
            )
        raise IntegrationNotConfiguredError(
            "petitRADTRANS adapter seam is present, but real backend wiring is " "not yet complete."
        )

    def transmission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Return a transmission spectrum for `theta` once the real backend is wired."""

        self._validate_backend()
        raise AssertionError("Unreachable after backend validation.")

    def emission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Return an emission spectrum for `theta` once the real backend is wired."""

        self._validate_backend()
        raise AssertionError("Unreachable after backend validation.")
