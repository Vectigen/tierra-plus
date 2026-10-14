# Conditional reliability audit

`tierra-plus audit` turns a finite spectral grid into an operational inference and
abstention report. It is meant for a user who already has model spectra, a labelled
calibration set, a separate labelled validation set and one unlabelled observation.
It does not call the deferred atmosphere backends.

```text
finite support ----+-------------------------------> observation inference
                   |                                      |
calibration states +--> choose chi-square threshold       |
                                                          v
validation states ----> coverage + acceptance gate --> accept / abstain
```

The roles must have distinct dataset IDs, source fingerprints, state IDs and
physical-state fingerprints. Calibration chooses the threshold. Validation alone
qualifies the resulting coverage--acceptance tradeoff. The observation has no
label and cannot affect either stage. These identity checks can catch declared
reuse; they cannot prove that producer-supplied metadata is truthful.

The command uses exact enumeration over the supplied finite support, marginalizes
a free additive transit-depth offset, and assumes known independent Gaussian error
bars. A result of `conditional_accept` means only that the validation point gates
passed, the observed minimum chi-square is below the calibration threshold, the
interval is below the declared width cap, and the interval does not touch the label
support boundary. It is not a guarantee against omitted clouds, chemistry,
correlated noise, instrument effects, stellar contamination or other model-family
changes. Discrete labels do not supply continuous physical precision.

## Input bundles

Each role is a directory containing `manifest.json` and `data.npz`. The manifest
uses schema version 1 and contains:

```json
{
  "schema_version": 1,
  "role": "support",
  "dataset_id": "unique-selection-id",
  "source": "human-readable source and selection",
  "source_fingerprint": "64 lowercase hex characters",
  "state_ids": ["one ID per physical state"],
  "state_fingerprints": ["one SHA-256 per physical state"],
  "data_file": "data.npz",
  "data_sha256": "SHA-256 of data.npz",
  "wavelength_unit": "micron",
  "spectral_quantity": "transit_depth"
}
```

The NPZ members are strict:

| Role | Arrays |
| --- | --- |
| support | `wavelength_microns[bins]`, `spectra[states, nuisance, bins]` (or 2D), `labels[states]` |
| calibration | `wavelength_microns[bins]`, `observations[states, bins]`, `sigma[bins]`, `labels[states]` |
| validation | the same arrays as calibration, from distinct physical states |
| observation | `wavelength_microns[bins]`, `observation[bins]`, `sigma[bins]` |

Wavelength centers must be positive, strictly increasing and byte-for-byte equal
across roles. The uncertainty vector must also match exactly across calibration,
validation and observation; version 1 does not silently rescale a threshold to a
new noise model. Units and spectral quantity are explicit. Archives and each NPY
header are bounded and hashed before NumPy allocates arrays; object or complex
arrays, duplicate or extra members, symbolic links and output replacement are
rejected. The report records both the data archive SHA-256 and the manifest
SHA-256 for every role, binding its byte payload to the declared identities.

Run the physical-asset demonstration from a source checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m examples.reliability_audit_demo --output /tmp/tierra-audit-demo
```

It prepares disjoint subsets of the already exposed PICASO RCTE grid, checks the
accept path on an explicitly interior-label panel, checks a photochemical
wrong-family spectrum, runs a separate panel with all 32 calibration and all 32
validation states, and verifies that shifted wavelength centers are rejected. The
interior-label panel was chosen after review added the support-boundary abstention;
it is an outcome-informed software-path example and cannot qualify the full RCTE
support. `demo-design.json` records that timeline, selected labels and panel sizes.
The full heldout report applies the unchanged operational rule including boundary
rejection. It accepts 22/32 validation states (68.75%) and therefore abstains below
the 70% floor, although all 22 selected discrete intervals cover. An earlier demo
that did not apply the boundary rule during validation is retained privately as
invalidated development evidence. All of these states and outcomes are exposed, so
none is fresh atmospheric evidence. The
[standalone four-case record](https://github.com/Vectigen/tierra-plus/blob/main/examples/results/reliability-audit/README.md)
preserves the exact gates, Wilson intervals and macOS reference-run output hashes.
Exact report bytes are expected only within the same numerical environment;
portable replay compares low-order floating metrics with absolute tolerance 1e-9
and relative tolerance 1e-12 while keeping decisions and counts exact.

To audit prepared bundles directly:

```bash
tierra-plus audit \
  --support /path/to/support \
  --calibration /path/to/calibration \
  --validation /path/to/validation \
  --observation /path/to/observation \
  --output /path/to/new-report.json
```

Defaults require at least 10 physical states in calibration and validation, set the
score threshold at the 90th calibration percentile, and qualify it only when
validation acceptance is at least 70% and accepted interval coverage is at least
85%. The validation selection applies the exact operational rule: score below the
threshold, interval no wider than 0.6 dex, and neither interval endpoint on the
label-support boundary. The report includes Wilson intervals to show the finite-state
uncertainty, while the gates use the declared point estimates.

The [IVOA Spectrum Data Model](https://www.ivoa.net/documents/SpectrumDM/) motivates
explicit spectral coordinates and associated metadata. The
[W3C PROV recommendation](https://www.w3.org/TR/prov-o/) motivates recording which
entities an activity used and generated. Split-conformal calibration has rigorous
coverage under exchangeability; the
[conformal prediction introduction](https://arxiv.org/abs/2107.07511) discusses that
assumption and distribution shift. This audit therefore reports empirical validation
under the supplied physical partition and makes no distribution-free shift claim.
