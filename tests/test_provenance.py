from __future__ import annotations

import io
import json
from pathlib import Path

from tierra_plus.models import BackendConfig
from tierra_plus.provenance.hash import content_hash, hash_path
from tierra_plus.provenance.log import RunLogEntry, append_run_log_entry, load_run_log_entries
from tierra_plus.provenance.replay import ReplayAsset, ReplayManifest


def test_content_hash_is_stable_for_json_payload_ordering() -> None:
    payload_a = {
        "backend": {"name": "tierra", "version": "1.0"},
        "priors": {"temperature_kelvin": 600.0, "metallicity": 0.2},
    }
    payload_b = {
        "priors": {"metallicity": 0.2, "temperature_kelvin": 600.0},
        "backend": {"version": "1.0", "name": "tierra"},
    }

    assert content_hash(payload_a) == content_hash(payload_b)

    assert hash_path(Path(__file__)) == content_hash(Path(__file__).read_bytes())


def test_run_log_appends_entries_without_overwriting() -> None:
    log_stream = io.StringIO()

    first = RunLogEntry(
        run_id="run-001",
        event="start",
        recorded_at_utc="2026-04-18T00:00:00Z",
        payload={"seed": 13},
    )
    second = RunLogEntry(
        run_id="run-002",
        event="finish",
        recorded_at_utc="2026-04-18T00:01:00Z",
        payload={"status": "ok"},
    )

    append_run_log_entry(log_stream, first)
    append_run_log_entry(log_stream, second)

    assert log_stream.getvalue().splitlines() == [
        json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":")),
        json.dumps(second.to_dict(), sort_keys=True, separators=(",", ":")),
    ]
    log_stream.seek(0)
    assert list(load_run_log_entries(log_stream)) == [first, second]


def test_replay_manifest_round_trips_through_dict_payload() -> None:
    backend = BackendConfig(
        backend_name="tierra",
        module_name="tierra.backend",
        data_root=Path("data/public"),
        extra={"mode": "transmission"},
    )
    manifest = ReplayManifest(
        run_id="run-abc123",
        seed=7,
        backend=backend,
        priors={"temperature_kelvin": 600.0, "cloud_top_pressure_bar": 0.1},
        assets=(
            ReplayAsset(
                identifier="synthetic-spectrum",
                kind="benchmark",
                sha256="abc123",
                source="data/public/benchmark/spectrum.json",
            ),
        ),
    )

    payload = manifest.to_dict()

    assert payload["seed"] == 7
    assert payload["backend"]["data_root"] == Path("data/public").as_posix()
    assert payload["assets"][0]["identifier"] == "synthetic-spectrum"
    assert ReplayManifest.from_dict(payload) == manifest
