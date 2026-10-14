# Sonora Diamondback photometry preparation

This record confirms bounded ingestion only. No temperature inference, threshold,
coverage, acceptance or cloud-challenge outcome has been computed.

On 2026-10-14, the preparation tool verified the live
[Zenodo record](https://zenodo.org/records/12735103) as DOI
`10.5281/zenodo.12735103`, license `CC-BY-4.0`, and `photometry.zip` identity
`md5:04e9e4a48883bab44b743d8feaaad759`, 244,314 bytes. The independently pinned
archive SHA-256 is
`7210f358c6da317d60a19dffb5b18a398e20a565aebaaa297e521ee9076bbc9c`.

The verified archive produced a 1,440-state, 16-band canonical grid with SHA-256
`ece152e796209f8a05803068c0f1849735d15d4e4f1224e74da265b209fdf51b`.
Coordinate rules selected 144 support, 48 calibration, 48 validation and 240 paired
cloud-challenge states. Those counts validate the frozen partition; they say
nothing about its eventual inference performance.

Reproduce from a checkout:

```bash
OUT=$(mktemp -d)
uv run python scripts/prepare_sonora_photometry.py \
  --verify-record \
  --output "$OUT/sonora-photometry-v1"
```

The [protocol](../../../docs/sonora-photometry-protocol.md) fixes the scientific
question, source choice, coordinate split, future synthetic noise and evaluation
gates. The machine-readable [record](record.json) binds this preparation evidence
to the exact protocol and adapter source.
