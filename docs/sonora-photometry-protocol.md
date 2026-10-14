# Prospective Sonora cloud-omission challenge

This protocol prepares a fresh, labelled missing-physics challenge without running
it. The question is narrow and falsifiable: **can a finite-grid temperature audit
calibrated only on cloud-free band photometry retain its declared selective
coverage and acceptance when refractory clouds are introduced?** A later negative
result would demonstrate failure for this grid. A later pass would remain limited
to this synthetic grid and noise model.

## Why this source

The 2024 Sonora Diamondback paper publishes 1,440 atmosphere models spanning
effective temperature 900--2400 K, log gravity 3.5--5.5, metallicity -0.5, 0 and
+0.5 dex, and cloud settings `fsed={1,2,3,4,8,nc}`. Its cloud model includes
refractory condensates and changes the resulting spectra and thermal evolution.
The authors explicitly describe the grid as applicable to brown dwarfs and directly
imaged planets, and publish the products openly. See
[Morley et al. (2024)](https://arxiv.org/html/2402.00758v1) and the
[versioned Zenodo record](https://zenodo.org/records/12735103).

The Zenodo record declares **CC BY 4.0**. Its full `spectra.zip` is 8.56 GB, larger
than this Mac's available 3.4 GiB, so the preparation tool refuses to treat that as
the active path. The official `photometry.zip` is 244,314 bytes and contains the
same labelled parameter grid integrated over 16 infrared bands. It is sufficient
for a band-photometry stress test and is not described as a full spectrum.

Two newer alternatives were not selected. The 2025 INARA release contains more
than three million synthetic terrestrial-exoplanet spectra, each with 15,346
spectral elements; it is attractive for later work but too large for the present
disk and time budget ([Zorzan et al. 2025](https://doi.org/10.3847/1538-4365/adb03a)).
The 2025 public K2-18 b joint analysis contains real spectra and retrieval products,
but its atmospheric labels are model-dependent rather than known simulation truth,
so it cannot directly measure coverage in this contract
([Luque et al. 2025 data](https://zenodo.org/records/15685551)).

## Frozen ingestion

The machine-readable protocol is
[`examples/configs/sonora-diamondback-photometry-v1.json`](https://github.com/Vectigen/tierra-plus/blob/main/examples/configs/sonora-diamondback-photometry-v1.json).
It was frozen before any inference or challenge metric was computed. It pins the
DOI, CC BY 4.0 license, archive URL, byte count, MD5 from Zenodo, an independent
SHA-256, the three selected member hashes, table schema, coordinate axes, partition
rules, future noise realization and gates.

Prepare the canonical grid from an already downloaded archive:

```bash
uv run python scripts/prepare_sonora_photometry.py \
  --archive /path/to/photometry.zip \
  --verify-record \
  --output /tmp/sonora-photometry-v1
```

Omit `--archive` to fetch the 244,314-byte file from the pinned Zenodo record. The
tool bounds the metadata response, compressed archive, member count, uncompressed
size and individual members; rejects symbolic links, duplicate names, encryption,
unsupported compression, malformed schemas, non-finite values and coordinate
changes; and verifies the live DOI, license, archive identity and selected-member
hashes. It writes a deterministic `grid.npz` and a provenance `manifest.json`
without replacing an existing output.

## Prospective partition

Cloud-free rows use `fsed=0` in the published table. Gravity is recorded in
m/s²; the five values map to log g 3.5--5.5 in cgs. Coordinate rules, rather than
shuffled row numbers or observed performance, define every role:

| Role | Cloud state | Gravity (m/s²) | States | Use |
| --- | --- | --- | ---: | --- |
| Support | `fsed=0` | 31.6, 316, 3160 | 144 | Temperature candidates with gravity and metallicity nuisance states |
| Calibration | `fsed=0` | 100 | 48 | Fit the score threshold only |
| Validation | `fsed=0` | 1000 | 48 | Qualify the complete cloud-free selection rule |
| Cloud challenge | `fsed=1,2,3,4,8` | 1000 | 240 | One untouched paired missing-cloud-physics test |

Every challenge coordinate is paired to a validation coordinate by temperature,
gravity and metallicity; only cloud sedimentation differs. The remaining cloudy
gravities are excluded and cannot be used for tuning.

The future run is also frozen: one NumPy seed sequence `[20261014, 2]`, independent
0.05-mag errors in all 16 bands, and a free additive grey offset. The target is the
16-level discrete effective temperature. Calibration uses the 90th score
percentile. Validation must reach at least 70% acceptance and 85% coverage among
accepted states, with intervals no wider than 300 K, before the cloud challenge can
ever conditionally accept. The cloud challenge is evaluated once under exactly the
same rule. No threshold, partition, noise scale, seed or gate may change after its
outcomes are visible.

## Claim boundary

No reliability result exists yet. This commit establishes ingestion and a
prospective partition only. Band-integrated synthetic magnitudes are not resolved
spectra or real observations. The additive offset is a grey magnitude nuisance,
not a radius or distance retrieval. Discrete 100 K labels do not imply continuous
temperature precision. The source assumes one model family, radiative-convective
and chemical equilibrium, and its own cloud parameterization.
