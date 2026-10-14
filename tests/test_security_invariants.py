"""Regression checks for provenance and numerical result-integrity boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts import bootstrap_public_data as bootstrap
from tierra_plus.forward.opacity import get_opacity_model
from tierra_plus.forward.opacity.ensemble import BenchmarkCase, posterior_weights_for_case
from tierra_plus.models import Spectrum
from tierra_plus.provenance.hash import content_hash, hash_bytes


def case_with_error(sigma: float, *, both_wrong: bool = False) -> BenchmarkCase:
    return BenchmarkCase(
        "integrity",
        "Finite inputs must not silently select a NaN winner",
        Spectrum([1], [1], [sigma]),
        {"wrong": Spectrum([1], [2]), "correct": Spectrum([1], [3 if both_wrong else 1])},
        "correct",
    )


def test_tiny_positive_uncertainty_does_not_select_first_nan_candidate():
    result = posterior_weights_for_case(case_with_error(1e-200))
    assert result.best_model == "correct"
    assert result.weights == {"wrong": 0.0, "correct": 1.0}
    assert np.isfinite(result.log_evidences["correct"])


def test_all_unrepresentable_candidate_likelihoods_fail_explicitly():
    with pytest.raises(ValueError, match="finite log evidence"):
        posterior_weights_for_case(case_with_error(1e-200, both_wrong=True))


@pytest.mark.parametrize(
    "priors", [{"wrong": 1e308, "correct": 1e308}, {"wrong": 1e-320, "correct": 1e308}]
)
def test_extreme_valid_priors_do_not_overflow_normalization(priors):
    result = posterior_weights_for_case(case_with_error(1), priors)
    assert all(np.isfinite(v) for v in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1)
    assert result.best_model == "correct"


def test_large_negative_equal_evidence_still_normalizes_weights():
    case = BenchmarkCase(
        "large",
        "Equal hypotheses",
        Spectrum([1], [0], [1]),
        {"a": Spectrum([1], [1e100]), "b": Spectrum([1], [1e100])},
        "a",
    )
    result = posterior_weights_for_case(case)
    assert result.weights == {"a": 0.5, "b": 0.5}


def test_large_negative_equal_likelihood_preserves_unequal_priors():
    case = BenchmarkCase(
        "large-priors",
        "Identical likelihoods must preserve prior odds",
        Spectrum([1], [0], [1]),
        {"a": Spectrum([1], [1e100]), "b": Spectrum([1], [1e100])},
        "b",
    )
    result = posterior_weights_for_case(case, {"a": 1, "b": 3})
    assert result.weights == pytest.approx({"a": 0.25, "b": 0.75})
    assert result.best_model == "b"


def test_large_finite_flux_difference_is_scaled_without_overflow():
    case = BenchmarkCase(
        "large-flux",
        "Representable standardized residuals",
        Spectrum([1], [1e308], [1e308]),
        {"a": Spectrum([1], [-1e308]), "b": Spectrum([1], [0])},
        "b",
    )
    result = posterior_weights_for_case(case)
    assert result.weights == pytest.approx({"a": 0.18242552380635635, "b": 0.8175744761936437})


@pytest.mark.parametrize(
    "payload", [{1: "value"}, {"nested": {True: 1}}, {"1": "value", 1: "different"}]
)
def test_provenance_rejects_keys_that_would_alias_after_string_coercion(payload):
    with pytest.raises(TypeError, match="string keys"):
        content_hash(payload)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_provenance_rejects_nonfinite_json_values(value):
    with pytest.raises(ValueError):
        content_hash({"result": value})


def downloadable_asset():
    return replace(
        get_opacity_model("synthetic-opacity-reference"),
        relative_path=Path("test/asset.bin"),
        download_url="https://example.invalid/asset.bin",
        sha256=hash_bytes(b"verified"),
    )


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_download_leaves_no_unverified_public_asset(tmp_path, monkeypatch, interrupted):
    asset = downloadable_asset()

    def download(url, destination):
        Path(destination).write_bytes(b"unverified partial bytes")
        if interrupted:
            raise OSError("interrupted transfer")

    monkeypatch.setattr(bootstrap.urllib.request, "urlretrieve", download)
    with pytest.raises((OSError, ValueError)):
        bootstrap.download_verified_asset(asset, tmp_path)
    assert not asset.resolve_path(tmp_path).exists()
    assert not list(tmp_path.rglob(".tierra-download-*"))


def test_verified_download_does_not_overwrite_concurrent_symlink(tmp_path, monkeypatch):
    asset = downloadable_asset()
    target = tmp_path / "untouched.bin"
    target.write_bytes(b"must remain unchanged")
    destination = asset.resolve_path(tmp_path)

    def download(url, temporary):
        Path(temporary).write_bytes(b"verified")
        destination.symlink_to(target)

    monkeypatch.setattr(bootstrap.urllib.request, "urlretrieve", download)
    with pytest.raises(FileExistsError):
        bootstrap.download_verified_asset(asset, tmp_path)
    assert target.read_bytes() == b"must remain unchanged"
    assert destination.is_symlink()
    assert not list(tmp_path.rglob(".tierra-download-*"))


def test_verified_download_publishes_exact_verified_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlretrieve",
        lambda url, destination: Path(destination).write_bytes(b"verified"),
    )
    path = bootstrap.download_verified_asset(downloadable_asset(), tmp_path)
    assert path.read_bytes() == b"verified"
    assert not list(tmp_path.rglob(".tierra-download-*"))
