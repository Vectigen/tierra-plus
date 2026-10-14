# Reliability audit: full-panel abstention and bounded software controls

The deployable audit's lead physical demonstration **abstains**. With all 32
exposed RCTE calibration states and all 32 exposed RCTE validation states, 22/32
validation states passed the exact operational selection rule: score below the
calibrated threshold, interval width at most 0.6 dex, and neither endpoint on the
support boundary. Acceptance was 68.75%, below the fixed 70% floor. All 22 selected
discrete-label intervals covered, but this does not rescue the failed acceptance
gate or turn grid labels into continuous metallicity precision.

These are already exposed PICASO states with synthetic noise. This is a software
demonstration and retrospective diagnostic, not fresh validation or evidence for
real observations. The 90% score threshold, 70% acceptance floor, 85% accepted
coverage floor and 0.6 dex width cap were unchanged across the four cases below.

| Case | Result | Evidence | Interpretation |
| --- | --- | --- | --- |
| Full-state qualification | `abstain` | 22/32 accepted; accepted coverage 22/22; threshold 145.558 | Lead negative result: acceptance misses its floor. |
| Interior-label software example | `conditional_accept` | 11/12 accepted; accepted coverage 11/11 | Outcome-informed panel restricted to labels {0.0, 0.5, 1.0}; valid only for exercising the accept code path. |
| Fixed photochemical observation | `abstain` | chi-square 394.239 > 146.914 and interval touches the lower support boundary | Prespecified exposed photochemical state index 0 under the interior-panel threshold. |
| Shifted wavelength geometry | rejected before inference | centers differ by 1e-6 micron | Exact observation geometry is mandatory. |

An earlier development demo omitted support-boundary abstention from validation
while applying it to the observation. It reported acceptance under an inconsistent
rule and is invalidated. After review fixed that contract, the interior-label panel
was chosen with outcome knowledge. It is retained and labelled because it is useful
for the user path, but it cannot support a general RCTE claim. The full 32/32 panel
was then run under the corrected unchanged rule and leads this result.

## Reproduce

From a source checkout:

```bash
AUDIT_DEMO=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m examples.reliability_audit_demo --output "$AUDIT_DEMO/run"
```

The command creates bounded, hashed support/calibration/validation/observation
bundles plus `full-heldout-report.json`, `interior-label-report.json`,
`wrong-family-report.json`, `controls.json` and `demo-design.json`. The NPZ writer
uses stable member order and timestamps. The recorded artifact digests preserve the
macOS reference run, and repeated runs in the same numerical environment produce
identical bytes. Different BLAS or platform implementations can change low-order
floating-point values and therefore report bytes. Portable replay keeps decisions,
counts, gates, reasons and identities exact, and compares floating metrics with
absolute tolerance 1e-9 and relative tolerance 1e-12. Every report records both
archive and manifest SHA-256 values.

The machine-readable [record](record.json) contains the gates, counts, Wilson
intervals, source-grid identities and output hashes. Read the
[input contract](../../../docs/reliability-audit.md) before replacing the demo
bundles with another physical grid.
