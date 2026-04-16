"""Command-line interface for Tierra+."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tierra_plus import __version__
from tierra_plus.exceptions import IntegrationNotConfiguredError
from tierra_plus.forward.prt_wrapper import PetitRadTransForwardModel
from tierra_plus.forward.tierra_wrapper import TierraForwardModel
from tierra_plus.models import AtmosphericState, BackendConfig


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
    return parser


def _build_backend(name: str) -> TierraForwardModel | PetitRadTransForwardModel:
    if name == "tierra":
        return TierraForwardModel(BackendConfig(backend_name="tierra", module_name="tierra"))
    return PetitRadTransForwardModel(
        BackendConfig(backend_name="petitRADTRANS", module_name="petitRADTRANS")
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
