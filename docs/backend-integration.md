# Backend Integration Contract

Tierra+ uses a fail-closed backend contract so unresolved scientific backends do not masquerade
as production-ready integrations.

## Current Status

| Backend | Status | Contract Notes |
| --- | --- | --- |
| `tierra` | canonical public repo identified, integration still blocked | The public upstream candidate is `disruptiveplanets/tierra`, but Tierra+ does not treat it as production-ready until a reproducible revision and data path are pinned in this repo. |
| `petitRADTRANS` | first live backend target | The official source of truth is the upstream GitLab project; Tierra+ records that source now, but live execution is still gated behind missing backend wiring and public-input verification. |

## Contract Shape

The backend contract records:

- backend name
- source kind: `submodule`, `vendored`, `pip`, or `git`
- source location
- source version
- installation status
- supported call modes

The current `tierra` placeholder contract is intentionally conservative: it records the public
source location, marks the version as unresolved, and only advertises `transmission` until
emission support is pinned and verified.

## Acceptance Criteria For An Upstream Backend

An upstream backend can be accepted into Tierra+ when all of the following are true:

- the source is public and traceable
- the source kind is explicit
- the source location is pinned to a repository URL, vendored snapshot, or package source
- the source version is pinned to a release tag, commit hash, or equivalent immutable identifier
- import and runtime checks fail closed with actionable errors if the backend is missing
- transmission and emission paths are covered by focused tests
- the backend does not silently fall back to a different scientific model

## Notes

This document describes the integration contract, not a scientific validation result. The package
still requires public-data-backed verification before any backend is treated as canonical.

The current public upstream references tracked by Tierra+ are:

- `tierra`: [https://github.com/disruptiveplanets/tierra](https://github.com/disruptiveplanets/tierra)
- `petitRADTRANS`: [https://gitlab.com/mauricemolli/petitRADTRANS](https://gitlab.com/mauricemolli/petitRADTRANS)
