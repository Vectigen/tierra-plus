"""Replay-manifest serialization for provenance-aware runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tierra_plus.models import BackendConfig


@dataclass(frozen=True)
class ReplayAsset:
    """One data asset referenced by a replay manifest."""

    identifier: str
    kind: str
    sha256: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the asset into JSON-compatible primitives."""

        return {
            "identifier": self.identifier,
            "kind": self.kind,
            "sha256": self.sha256,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayAsset:
        """Construct a replay asset from a mapping payload."""

        return cls(
            identifier=str(payload["identifier"]),
            kind=str(payload["kind"]),
            sha256=str(payload["sha256"]),
            source=str(payload["source"]),
        )


def _backend_config_to_dict(config: BackendConfig) -> dict[str, Any]:
    return config.to_dict()


def _backend_config_from_dict(payload: Mapping[str, Any]) -> BackendConfig:
    return BackendConfig.from_dict(payload)


@dataclass(frozen=True)
class ReplayManifest:
    """Compact manifest needed to replay one provenance-aware run."""

    run_id: str
    seed: int
    backend: BackendConfig
    priors: Mapping[str, float]
    assets: tuple[ReplayAsset, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest into JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "backend": _backend_config_to_dict(self.backend),
            "priors": dict(self.priors),
            "assets": [asset.to_dict() for asset in self.assets],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayManifest:
        """Construct a replay manifest from a mapping payload."""

        return cls(
            run_id=str(payload["run_id"]),
            seed=int(payload["seed"]),
            backend=_backend_config_from_dict(payload["backend"]),
            priors={str(key): float(value) for key, value in dict(payload["priors"]).items()},
            assets=tuple(ReplayAsset.from_dict(entry) for entry in payload["assets"]),
        )
