# Synthetic coverage check

The separate [finite-grid reliability audit record](reliability-audit/README.md)
contains the operational four-case demonstration and its negative full-panel result.
The [Sonora preparation record](sonora-photometry-preparation/README.md) separately
freezes a verified CC-BY-4.0 ingestion and prospective cloud-omission partition;
the subsequent [one-run evaluation](sonora-photometry-evaluation/README.md) qualified
on the cloud-free validation split and failed both acceptance and accepted-coverage
gates on the paired cloudy challenge.

These are measurements from 20 completed local runs of `examples/coverage_check.py`
with noise seeds 1000 through 1019 and 90% weighted equal-tail posterior intervals.
Each run uses 150 live points and `dlogz=0.05`. The two true parameters are fixed;
only observation noise and sampler randomness vary. This is a software regression
check, not prior-predictive simulation-based calibration or atmospheric validation.

| Parameter | Truth inside interval | Empirical coverage | Wilson 95% interval |
| --- | --- | --- | --- |
| Baseline depth | 18 / 20 | 0.90 | [0.6990, 0.9721] |
| Feature amplitude | 16 / 20 | 0.80 | [0.5840, 0.9193] |

The second parameter under-covers in this small run. Its binomial uncertainty is
wide; this does not establish either a calibrated retrieval system or a systematic
coverage defect. The per-fit analytic Gaussian comparison is retained in the raw
records. Larger repeated-noise and prior-predictive checks, correlated noise,
model misspecification and real radiative-transfer integrations remain necessary.

Reproduce from the repository root:

```sh
uv sync --locked --dev --extra inference
uv run python examples/coverage_check.py --runs 20 --seed 1000 --level 0.9 --output results/coverage
```

`coverage-seed1000.json.gz` contains the complete summary and all 20 fit records,
including observations, priors, sampler configuration, weighted posterior samples,
package versions, input hashes and analytic checks. The output directory must not
already exist, to avoid silently overwriting an experiment.

SHA-256 of the compressed artifact: `7e7b4aa4aafad9f143b90556ef256bdd64b24ff0c609d5e59087cf57e97978b3`.
