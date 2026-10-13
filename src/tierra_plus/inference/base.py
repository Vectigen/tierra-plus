"""Core sampler abstractions for Tierra+."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplerResult:
    """Serializable output from a sampling run.

    Parameters
    ----------
    parameter_names
        Ordered parameter names corresponding to each sample column.
    samples
        Weighted nested-sampling points in parameter order; use ``weights``.
    log_likelihoods
        Log-likelihood values aligned with ``samples``.
    log_evidence
        Final log-evidence estimate reported by the sampler.
    weights
        Normalized posterior weights aligned with samples.
    log_evidence_error
        Estimated standard error of the log evidence.
    metadata
        Additional sampler metadata that should remain JSON-compatible.
    """

    parameter_names: tuple[str, ...]
    samples: tuple[tuple[float, ...], ...]
    log_likelihoods: tuple[float, ...]
    log_evidence: float
    weights: tuple[float, ...]
    log_evidence_error: float
    metadata: Mapping[str, object] = field(default_factory=dict)


class Sampler(ABC):
    """Abstract sampler interface for baseline retrieval lanes."""

    @abstractmethod
    def run(self) -> SamplerResult:
        """Execute the sampler and return a typed result payload."""
