"""Adapter seam for the upstream `tierra` forward model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.base import ForwardModel
from tierra_plus.models import AtmosphericState, Spectrum


class TierraForwardModel(ForwardModel):
    """Wrap the unresolved upstream `tierra` backend behind the stable interface.

    Notes
    -----
    The package intentionally exposes this seam before the canonical upstream source is selected.
    Until that dependency is resolved, calls fail with `IntegrationNotConfiguredError` instead of
    pretending a scientific backend is available.
    """

    def _validate_backend(self) -> None:
        module_name = self.config.module_name or "tierra"
        if importlib.util.find_spec(module_name) is None:
            raise IntegrationNotConfiguredError(
                f"tierra backend module '{module_name}' is not installed. "
                "Resolve the canonical upstream source and vendor it before use."
            )
        if self.config.data_root is not None and not Path(self.config.data_root).exists():
            raise IntegrationNotConfiguredError(
                f"Configured tierra data root '{self.config.data_root}' does not exist."
            )
        raise IntegrationNotConfiguredError(
            "tierra adapter seam is present, but real backend wiring is intentionally unresolved."
        )

    def transmission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Return a transmission spectrum for `theta` once the real backend is wired."""

        self._validate_backend()
        raise AssertionError("Unreachable after backend validation.")

    def emission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Return an emission spectrum for `theta` once the real backend is wired."""

        self._validate_backend()
        raise AssertionError("Unreachable after backend validation.")
