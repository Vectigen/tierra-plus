from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_bootstrap_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_public_data.py"
    spec = importlib.util.spec_from_file_location("bootstrap_public_data", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load bootstrap_public_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_verify_only_succeeds_for_tracked_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_public_data = load_bootstrap_module()
    monkeypatch.setattr(sys, "argv", ["bootstrap_public_data.py", "--verify-only"])

    assert bootstrap_public_data.main() == 0
