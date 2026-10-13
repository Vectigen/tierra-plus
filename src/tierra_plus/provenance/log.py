"""Append-only JSONL run logging primitives."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class RunLogEntry:
    """One append-only provenance event recorded in a JSONL run log."""

    run_id: str
    event: str
    recorded_at_utc: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry into JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "event": self.event,
            "recorded_at_utc": self.recorded_at_utc,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunLogEntry:
        """Construct a run-log entry from a mapping payload."""

        return cls(
            run_id=str(payload["run_id"]),
            event=str(payload["event"]),
            recorded_at_utc=str(payload["recorded_at_utc"]),
            payload=dict(payload["payload"]),
        )


def append_run_log_entry(handle: TextIO, entry: RunLogEntry) -> None:
    """Append one run-log entry as a JSON line."""

    handle.write(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def load_run_log_entries(handle: TextIO) -> Iterator[RunLogEntry]:
    """Iterate over JSONL run-log entries from a text stream."""

    for line in handle:
        if not line.strip():
            continue
        yield RunLogEntry.from_dict(json.loads(line))
