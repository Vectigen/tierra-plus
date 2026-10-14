"""Stable content hashing helpers for provenance artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _canonicalize_payload(payload: Any) -> Any:
    """Convert a JSON-compatible payload into a stable canonical structure."""

    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, bytes):
        return {
            "__type__": "bytes",
            "value": base64.b64encode(payload).decode("ascii"),
        }
    if isinstance(payload, bytearray):
        return _canonicalize_payload(bytes(payload))
    if isinstance(payload, memoryview):
        return _canonicalize_payload(payload.tobytes())
    if isinstance(payload, Mapping):
        if any(not isinstance(key, str) for key in payload):
            raise TypeError("Provenance mappings must use JSON string keys")
        return {
            "__type__": "mapping",
            "items": [
                [key, _canonicalize_payload(value)] for key, value in sorted(payload.items())
            ],
        }
    if isinstance(payload, (list, tuple)):
        return {
            "__type__": "sequence",
            "items": [_canonicalize_payload(value) for value in payload],
        }
    raise TypeError(f"Unsupported payload type for hashing: {type(payload)!r}")


def hash_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def hash_path(path: Path) -> str:
    """Return the SHA-256 digest for the bytes stored at ``path``."""

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def content_hash(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible provenance payloads.

    The helper accepts raw bytes, nested JSON-compatible mappings with string keys,
    and sequences. Nonfinite numbers and non-string keys are rejected rather than
    silently merging distinct provenance identities. Raw bytes retain ordinary
    file-SHA256 semantics; callers must keep file and structured-payload identities
    in separate namespaces. This is a digest, not an authenticity signature.

    Parameters
    ----------
    payload
        Bytes or JSON-compatible provenance content.
    """

    if isinstance(payload, bytes):
        return hash_bytes(payload)
    canonical = _canonicalize_payload(payload)
    serialized = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
