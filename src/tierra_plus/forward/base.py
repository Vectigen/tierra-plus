"""Base forward-model interface for Tierra+."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tierra_plus.models import AtmosphericState, BackendConfig, Spectrum


class ForwardModel(ABC):
    """Abstract forward-model interface used by Tierra+.

    Parameters
    ----------
    config
        Backend configuration describing where the concrete forward model lives.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    def transmission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Compute a transmission spectrum for `theta`."""

    @abstractmethod
    def emission_spectrum(self, theta: AtmosphericState) -> Spectrum:
        """Compute an emission spectrum for `theta`."""
