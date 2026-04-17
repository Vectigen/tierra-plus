from __future__ import annotations

import pytest

from tierra_plus.cli import (
    PETITRADTRANS_SOURCE_URL,
    TIERRA_SOURCE_URL,
    _build_backend,
    main,
)


def test_cli_version_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])

    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_retrieve_reports_unconfigured_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "retrieve",
            "--backend",
            "tierra",
            "--mode",
            "transmission",
            "--temperature-k",
            "500",
            "--radius-rjup",
            "0.25",
            "--gravity-ms2",
            "9.8",
            "--composition-json",
            '{"H2O": 0.2, "N2": 0.8}',
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not configured" in captured.err


def test_cli_backend_defaults_include_public_source_metadata() -> None:
    tierra = _build_backend("tierra")
    petitradtrans = _build_backend("petitradtrans")

    assert tierra.config.source_kind == "git"
    assert tierra.config.source_location == TIERRA_SOURCE_URL
    assert tierra.config.install_status == "not_installed"
    assert petitradtrans.config.source_kind == "pip"
    assert petitradtrans.config.source_location == PETITRADTRANS_SOURCE_URL
    assert petitradtrans.config.install_status == "not_installed"
