# Backend Integration Contract

Tierra+ uses a fail-closed backend contract so unresolved scientific backends do not masquerade
as production-ready integrations.

## Current Status

| Backend | Status | Contract Notes |
| --- | --- | --- |
| `tierra` | unresolved upstream selection | Contract metadata exists, but the source location and version remain unknown until a canonical upstream is pinned. |
| `petitRADTRANS` | adapter seam only | The package exposes a stable wrapper interface, but live execution is still gated behind missing backend wiring. |

## Contract Shape

The backend contract records:

- backend name
- source kind: `submodule`, `vendored`, `pip`, or `git`
- source location
- source version
- installation status
- supported call modes

The current Tierra placeholder contract supports both `transmission` and `emission` call modes and
raises an explicit `IntegrationNotConfiguredError` when source identity is still unknown.

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
