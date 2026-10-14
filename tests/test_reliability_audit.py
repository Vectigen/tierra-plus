from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tierra_plus.cli import main
from tierra_plus.provenance import hash_path
from tierra_plus.reliability import AuditCriteria, run_reliability_audit


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bundle(
    root: Path,
    role: str,
    arrays: dict[str, np.ndarray],
    states: int,
    *,
    fingerprint_prefix: str | None = None,
) -> Path:
    path = root / role
    path.mkdir()
    np.savez_compressed(path / "data.npz", **arrays)
    prefix = fingerprint_prefix or role
    manifest = {
        "schema_version": 1,
        "role": role,
        "dataset_id": f"fixture-{role}",
        "source": "synthetic test fixture",
        "source_fingerprint": _digest(f"source-{role}"),
        "state_ids": [f"{role}-{index}" for index in range(states)],
        "state_fingerprints": [_digest(f"{prefix}-{index}") for index in range(states)],
        "data_file": "data.npz",
        "data_sha256": hash_path(path / "data.npz"),
        "wavelength_unit": "micron",
        "spectral_quantity": "transit_depth",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _audit_bundles(root: Path, *, bad_validation: bool = False) -> tuple[Path, ...]:
    wave = np.linspace(1.0, 2.0, 5)
    shape = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 0.1
    spectra = np.stack([shape, np.zeros(5), -shape])[:, None, :]
    sigma = np.full(5, 0.01)
    rng = np.random.default_rng(44)
    calibration = rng.normal(0, 1.2 * sigma, size=(12, 5))
    validation = rng.normal(0, 0.8 * sigma, size=(12, 5))
    if bad_validation:
        validation += np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    return (
        _bundle(
            root,
            "support",
            {"wavelength_microns": wave, "spectra": spectra, "labels": np.array([-1, 0, 1])},
            3,
        ),
        _bundle(
            root,
            "calibration",
            {
                "wavelength_microns": wave,
                "observations": calibration,
                "sigma": sigma,
                "labels": np.zeros(12),
            },
            12,
        ),
        _bundle(
            root,
            "validation",
            {
                "wavelength_microns": wave,
                "observations": validation,
                "sigma": sigma,
                "labels": np.zeros(12),
            },
            12,
        ),
        _bundle(
            root,
            "observation",
            {"wavelength_microns": wave, "observation": np.zeros(5), "sigma": sigma},
            1,
        ),
    )


def _refresh_hash(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_sha256"] = hash_path(bundle / "data.npz")
    manifest_path.write_text(json.dumps(manifest))


def test_audit_conditionally_accepts_only_after_disjoint_validation(tmp_path: Path) -> None:
    report = run_reliability_audit(*_audit_bundles(tmp_path))

    assert report["decision"] == "conditional_accept"
    assert report["reasons"] == []
    assert report["score_calibration"]["qualified"]
    assert report["score_calibration"]["validation"]["accepted_coverage"] == 1.0
    assert report["observation"]["mean"] == pytest.approx(0)
    assert report["observation"]["support_boundary"] is False
    assert "no generic OOD" in report["claim_boundary"]
    assert len(report["inputs"]["support"]["data_sha256"]) == 64
    assert len(report["inputs"]["support"]["manifest_sha256"]) == 64


def test_validation_failure_forces_abstention_even_when_observation_fits(tmp_path: Path) -> None:
    report = run_reliability_audit(*_audit_bundles(tmp_path, bad_validation=True))

    assert report["decision"] == "abstain"
    assert "validation_acceptance_below_floor" in report["reasons"]
    assert report["observation"]["score_pass"]


def test_wrong_family_observation_forces_abstention(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    wave = np.linspace(1.0, 2.0, 5)
    np.savez_compressed(
        observation / "data.npz",
        wavelength_microns=wave,
        observation=np.array([0.5, -0.5, 0.5, -0.5, 0.5]),
        sigma=np.full(5, 0.01),
    )
    _refresh_hash(observation)

    report = run_reliability_audit(*paths)

    assert report["decision"] == "abstain"
    assert "observation_score_above_calibrated_threshold" in report["reasons"]


def test_audit_rejects_declared_state_reuse_across_roles(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    calibration = json.loads((paths[1] / "manifest.json").read_text())
    validation_path = paths[2] / "manifest.json"
    validation = json.loads(validation_path.read_text())
    validation["state_fingerprints"][0] = calibration["state_fingerprints"][0]
    validation_path.write_text(json.dumps(validation))

    with pytest.raises(ValueError, match="physical-state fingerprint"):
        run_reliability_audit(*paths)


def test_audit_requires_sha256_state_fingerprints(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    manifest_path = paths[2] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state_fingerprints"][0] = "declared-but-not-hashed"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="state_fingerprints must be lowercase SHA-256"):
        run_reliability_audit(*paths)


def test_audit_rejects_geometry_mismatch_before_inference(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    np.savez_compressed(
        observation / "data.npz",
        wavelength_microns=np.linspace(1.0, 2.1, 5),
        observation=np.zeros(5),
        sigma=np.full(5, 0.01),
    )
    _refresh_hash(observation)

    with pytest.raises(ValueError, match="wavelength centers differ"):
        run_reliability_audit(*paths)


def test_audit_rejects_unexpected_npz_members_before_loading(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    np.savez_compressed(
        observation / "data.npz",
        wavelength_microns=np.linspace(1.0, 2.0, 5),
        observation=np.zeros(5),
        sigma=np.full(5, 0.01),
        surprise=np.zeros(1),
    )
    _refresh_hash(observation)

    with pytest.raises(ValueError, match="archive members"):
        run_reliability_audit(*paths)


def test_audit_rejects_bundle_symlink_before_resolving_it(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    link = tmp_path / "support-link"
    link.symlink_to(paths[0], target_is_directory=True)

    with pytest.raises(ValueError, match="bundle must not be a symbolic link"):
        run_reliability_audit(link, *paths[1:])


@pytest.mark.parametrize("corruption", ["duplicate", "forged_shape", "tiny_member"])
def test_audit_preflights_every_npy_member_before_allocation(
    tmp_path: Path, corruption: str
) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    archive_path = observation / "data.npz"
    if corruption == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("sigma.npy", b"duplicate")
        expected = "archive members"
    else:
        wave = np.linspace(1.0, 2.0, 5)
        sigma = np.full(5, 0.01)
        buffers: dict[str, bytes] = {}
        for name, value in (("wavelength_microns", wave), ("sigma", sigma)):
            handle = io.BytesIO()
            np.save(handle, value, allow_pickle=False)
            buffers[name] = handle.getvalue()
        if corruption == "forged_shape":
            handle = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                handle,
                {
                    "descr": np.dtype(np.float64).str,
                    "fortran_order": False,
                    "shape": (100_000_000,),
                },
            )
            buffers["observation"] = handle.getvalue()
            expected = "NPY shape|in-memory size"
        else:
            buffers["observation"] = b"tiny"
            expected = "invalid NPY header"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in buffers.items():
                archive.writestr(f"{name}.npy", value)
    _refresh_hash(observation)

    with pytest.raises(ValueError, match=expected):
        run_reliability_audit(*paths)


def test_audit_rejects_changed_uncertainty_geometry(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    np.savez_compressed(
        observation / "data.npz",
        wavelength_microns=np.linspace(1.0, 2.0, 5),
        observation=np.zeros(5),
        sigma=np.full(5, 0.02),
    )
    _refresh_hash(observation)

    with pytest.raises(ValueError, match="uncertainty geometry differs"):
        run_reliability_audit(*paths)


def test_audit_fails_closed_on_extreme_finite_uncertainties(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    for bundle in paths[1:]:
        archive = np.load(bundle / "data.npz")
        arrays = {key: archive[key] for key in archive.files}
        archive.close()
        arrays["sigma"] = np.full(5, 1e-300)
        np.savez_compressed(bundle / "data.npz", **arrays)
        _refresh_hash(bundle)

    with pytest.raises(ValueError, match="calibration inference is numerically invalid"):
        run_reliability_audit(*paths)


def test_audit_rejects_scalar_primary_array_with_value_error(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    np.savez_compressed(
        observation / "data.npz",
        wavelength_microns=np.linspace(1.0, 2.0, 5),
        observation=np.array(0.0),
        sigma=np.full(5, 0.01),
    )
    _refresh_hash(observation)

    with pytest.raises(ValueError, match="must not be scalar"):
        run_reliability_audit(*paths)


def test_audit_rejects_unsupported_zip_compression(tmp_path: Path) -> None:
    paths = _audit_bundles(tmp_path)
    observation = paths[-1]
    arrays = {
        "wavelength_microns": np.linspace(1.0, 2.0, 5),
        "observation": np.zeros(5),
        "sigma": np.full(5, 0.01),
    }
    with zipfile.ZipFile(observation / "data.npz", "w", compression=zipfile.ZIP_BZIP2) as archive:
        for name, value in arrays.items():
            handle = io.BytesIO()
            np.save(handle, value, allow_pickle=False)
            archive.writestr(f"{name}.npy", handle.getvalue())
    _refresh_hash(observation)

    with pytest.raises(ValueError, match="safety bounds"):
        run_reliability_audit(*paths)


def test_validation_uses_same_support_boundary_abstention_as_observation(
    tmp_path: Path,
) -> None:
    paths = _audit_bundles(tmp_path)
    validation = paths[2]
    wave = np.linspace(1.0, 2.0, 5)
    shape = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 0.1
    np.savez_compressed(
        validation / "data.npz",
        wavelength_microns=wave,
        observations=np.repeat(shape[None, :], 12, axis=0),
        sigma=np.full(5, 0.01),
        labels=np.full(12, -1.0),
    )
    _refresh_hash(validation)

    report = run_reliability_audit(*paths)

    assert report["decision"] == "abstain"
    assert report["score_calibration"]["validation"]["boundary_accepted"] == 0
    assert "validation_acceptance_below_floor" in report["reasons"]


def test_cli_writes_once_and_reports_decision(tmp_path: Path, capsys) -> None:
    paths = _audit_bundles(tmp_path)
    output = tmp_path / "report.json"
    arguments = [
        "audit",
        "--support",
        str(paths[0]),
        "--calibration",
        str(paths[1]),
        "--validation",
        str(paths[2]),
        "--observation",
        str(paths[3]),
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    assert json.loads(output.read_text())["decision"] == "conditional_accept"
    assert json.loads(capsys.readouterr().out)["decision"] == "conditional_accept"
    assert main(arguments) == 2
    assert "Refusing to replace" in capsys.readouterr().err


def test_criteria_reject_invalid_gates() -> None:
    with pytest.raises(ValueError, match="score_quantile"):
        AuditCriteria(score_quantile=1.0)


def _assert_recorded_numeric_metrics(
    reports: dict[str, dict[str, object]], record: dict[str, object]
) -> None:
    """Compare public metrics while allowing only platform-scale rounding."""
    tolerance = {"abs": 1e-9, "rel": 1e-12}
    cases = record["cases"]
    assert isinstance(cases, dict)
    full_case = cases["full_state_qualification"]
    interior_case = cases["interior_label_software_example"]
    wrong_case = cases["fixed_photochemical_observation"]
    assert isinstance(full_case, dict)
    assert isinstance(interior_case, dict)
    assert isinstance(wrong_case, dict)
    full_score = reports["full"]["score_calibration"]
    interior_score = reports["interior"]["score_calibration"]
    wrong_observation = reports["wrong"]["observation"]
    assert isinstance(full_score, dict)
    assert isinstance(interior_score, dict)
    assert isinstance(wrong_observation, dict)
    full_validation = full_score["validation"]
    interior_validation = interior_score["validation"]
    assert isinstance(full_validation, dict)
    assert isinstance(interior_validation, dict)
    for key in ("acceptance", "acceptance_wilson95", "accepted_coverage"):
        assert full_validation[key] == pytest.approx(full_case[key], **tolerance)
        assert interior_validation[key] == pytest.approx(interior_case[key], **tolerance)
    assert full_validation["accepted_coverage_wilson95"] == pytest.approx(
        full_case["accepted_coverage_wilson95"], **tolerance
    )
    assert interior_validation["accepted_coverage_wilson95"] == pytest.approx(
        interior_case["accepted_coverage_wilson95"], **tolerance
    )
    assert full_score["threshold"] == pytest.approx(full_case["score_threshold"], **tolerance)
    assert wrong_observation["minimum_chi2"] == pytest.approx(
        wrong_case["minimum_chi2"], **tolerance
    )
    assert wrong_observation["score_threshold"] == pytest.approx(
        wrong_case["score_threshold"], **tolerance
    )


def test_physical_demo_replays_public_four_case_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from examples import reliability_audit_demo as demo

    output = tmp_path / "physical-demo"
    demo.run(output)
    summary = json.loads(capsys.readouterr().out)
    record_path = Path(__file__).resolve().parents[1] / "examples/results/reliability-audit"
    record = json.loads((record_path / "record.json").read_text())

    assert record["artifact_hash_scope"].startswith("macOS reference run")
    assert record["artifacts"] == {
        "controls.json": "32f0a5c29bb5938110c4c4cdd14db5c20d280b4b6ce3ed0a5db9b4dfdbffe006",
        "demo-design.json": "285dd321dcf34bd5c1a0644adf935ce266a8ec186cd1f59f3328b8540bfd716f",
        "full-heldout-report.json": (
            "bac1f9d39eae686d7472a3436b02b88b3a93663a40b328b30309f18171398bce"
        ),
        "interior-label-report.json": (
            "bfc9614402708536f0e1d7473915eac5c787652b44f9c9a232bf59676af7ea9c"
        ),
        "wrong-family-report.json": (
            "acfaae944edbf10993af7cb6f567c49ee62b3f397cba6d471acaaa0059db0920"
        ),
    }
    assert hash_path(Path(demo.__file__)) == record["writer_source_sha256"]
    method = Path(__file__).resolve().parents[1] / "src/tierra_plus/reliability.py"
    assert hash_path(method) == record["method_source_sha256"]
    assert summary == {
        "full_heldout": "abstain",
        "geometry_mismatch": "rejected",
        "interior_software_example": "conditional_accept",
        "output": str(output),
        "photochemical_observation": "abstain",
    }

    # Artifact bytes are deterministic within one numerical environment. Across
    # BLAS/platform implementations, low-order reductions can change JSON floats.
    replay = tmp_path / "same-host-replay"
    demo.run(replay)
    capsys.readouterr()
    for name in record["artifacts"]:
        assert hash_path(output / name) == hash_path(replay / name)

    full = json.loads((output / "full-heldout-report.json").read_text())
    full_case = record["cases"]["full_state_qualification"]
    full_validation = full["score_calibration"]["validation"]
    assert full["criteria"] == record["criteria"]
    assert full["decision"] == full_case["decision"]
    assert full["reasons"] == [
        "validation_acceptance_below_floor",
        "observation_score_above_calibrated_threshold",
        "observation_interval_touches_support_boundary",
    ]
    assert full["score_calibration"]["qualified"] is False
    assert full["score_calibration"]["qualification_failures"] == [full_case["reason"]]
    assert full_validation["states"] == full_case["validation_states"]
    assert full_validation["accepted"] == full_case["accepted"]
    assert full_validation["accepted_covered"] == full_case["accepted"]
    assert full_validation["score_accepted"] == 30
    assert full_validation["width_accepted"] == 32
    assert full_validation["boundary_accepted"] == 24

    interior = json.loads((output / "interior-label-report.json").read_text())
    interior_case = record["cases"]["interior_label_software_example"]
    interior_validation = interior["score_calibration"]["validation"]
    assert interior["criteria"] == record["criteria"]
    assert interior["decision"] == interior_case["decision"]
    assert interior["reasons"] == []
    assert interior["score_calibration"]["qualified"] is True
    assert interior_validation["states"] == interior_case["validation_states"]
    assert interior_validation["accepted"] == interior_case["accepted"]
    assert interior_validation["accepted_covered"] == interior_case["accepted"]
    assert interior["demonstration_context"]["status"] == interior_case["status"]

    wrong = json.loads((output / "wrong-family-report.json").read_text())
    wrong_case = record["cases"]["fixed_photochemical_observation"]
    assert wrong["criteria"] == record["criteria"]
    assert wrong["decision"] == wrong_case["decision"]
    assert wrong["reasons"] == wrong_case["reasons"]
    assert wrong["demonstration_context"]["observation"].endswith(
        f"index {wrong_case['state_index']}"
    )

    reports = {"full": full, "interior": interior, "wrong": wrong}
    _assert_recorded_numeric_metrics(reports, record)
    identities = {
        value["dataset_id"]: value
        for report in reports.values()
        for value in report["inputs"].values()
    }
    assert identities == record["input_identities"]

    changed_threshold = copy.deepcopy(reports)
    changed_threshold["full"]["score_calibration"]["threshold"] += 1e-4
    with pytest.raises(AssertionError):
        _assert_recorded_numeric_metrics(changed_threshold, record)
    changed_coverage = copy.deepcopy(reports)
    changed_coverage["full"]["score_calibration"]["validation"]["accepted_coverage"] -= 0.01
    with pytest.raises(AssertionError):
        _assert_recorded_numeric_metrics(changed_coverage, record)

    controls = json.loads((output / "controls.json").read_text())
    geometry_case = record["cases"]["geometry_mismatch"]
    assert controls["geometry_mismatch"] == {
        "rejected": True,
        "reason": geometry_case["reason"],
    }
    design = json.loads((output / "demo-design.json").read_text())
    assert design["full_panel"] == {
        "calibration_states": 32,
        "validation_states": 32,
        "observation": "prespecified exposed photochemical state index 0",
    }
    assert design["interior_label_panel"]["calibration_labels"] == interior_case["labels"]
    assert design["interior_label_panel"]["validation_labels"] == interior_case["labels"]
