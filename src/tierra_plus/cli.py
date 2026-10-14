"""Command-line interface for Tierra+."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tierra_plus import __version__
from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.prt_wrapper import PetitRadTransForwardModel
from tierra_plus.forward.tierra_wrapper import TierraForwardModel
from tierra_plus.models import AtmosphericState, BackendConfig
from tierra_plus.reliability import AuditCriteria, run_reliability_audit, write_audit_report

TIERRA_SOURCE_URL = "https://github.com/disruptiveplanets/tierra"
PETITRADTRANS_SOURCE_URL = "https://gitlab.com/mauricemolli/petitRADTRANS"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the Tierra+ CLI."""

    parser = argparse.ArgumentParser(prog="tierra-plus")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    retrieve = subparsers.add_parser("retrieve", help="Run a retrieval backend for one spectrum.")
    retrieve.add_argument("--backend", choices=("tierra", "petitradtrans"), required=True)
    retrieve.add_argument("--mode", choices=("transmission", "emission"), required=True)
    retrieve.add_argument("--temperature-k", type=float, required=True)
    retrieve.add_argument("--radius-rjup", type=float, required=True)
    retrieve.add_argument("--gravity-ms2", type=float, required=True)
    retrieve.add_argument("--composition-json", required=True)

    audit = subparsers.add_parser(
        "audit", help="Screen one spectrum with a calibrated finite-grid reliability contract."
    )
    audit.add_argument("--support", type=Path, required=True)
    audit.add_argument("--calibration", type=Path, required=True)
    audit.add_argument("--validation", type=Path, required=True)
    audit.add_argument("--observation", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--score-quantile", type=float, default=0.9)
    audit.add_argument("--minimum-states", type=int, default=10)
    audit.add_argument("--minimum-acceptance", type=float, default=0.7)
    audit.add_argument("--minimum-accepted-coverage", type=float, default=0.85)
    audit.add_argument("--maximum-interval-width", type=float, default=0.6)
    return parser


def _build_backend(name: str) -> TierraForwardModel | PetitRadTransForwardModel:
    if name == "tierra":
        return TierraForwardModel(
            BackendConfig(
                backend_name="tierra",
                module_name="tierra",
                source_kind="git",
                source_location=TIERRA_SOURCE_URL,
                install_status="not_installed",
            )
        )
    return PetitRadTransForwardModel(
        BackendConfig(
            backend_name="petitRADTRANS",
            module_name="petitRADTRANS",
            source_kind="pip",
            source_location=PETITRADTRANS_SOURCE_URL,
            install_status="not_installed",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Tierra+ CLI.

    Parameters
    ----------
    argv
        Optional argument vector for testing or embedding.

    Returns
    -------
    int
        Process exit code. A non-zero exit indicates validation or configuration failure.
    """

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "audit":
        try:
            criteria = AuditCriteria(
                score_quantile=args.score_quantile,
                minimum_states=args.minimum_states,
                minimum_acceptance=args.minimum_acceptance,
                minimum_accepted_coverage=args.minimum_accepted_coverage,
                maximum_interval_width=args.maximum_interval_width,
            )
            report = run_reliability_audit(
                args.support,
                args.calibration,
                args.validation,
                args.observation,
                criteria,
            )
            write_audit_report(report, args.output)
        except (OSError, ValueError) as error:
            print(f"Tierra+ audit rejected input: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"decision": report["decision"], "output": str(args.output)}))
        return 0
    composition = json.loads(args.composition_json)
    state = AtmosphericState(
        temperature_kelvin=args.temperature_k,
        radius_rjup=args.radius_rjup,
        gravity_m_s2=args.gravity_ms2,
        composition=composition,
    )
    model = _build_backend(args.backend)
    try:
        if args.mode == "transmission":
            model.transmission_spectrum(state)
        else:
            model.emission_spectrum(state)
    except IntegrationNotConfiguredError as error:
        print(f"Tierra+ backend not configured: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
