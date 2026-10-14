# Prospective Sonora cloud-omission result

The one frozen-seed run produced a **negative result**. The cloud-free validation
qualified under the predeclared point-estimate gates, so the evaluator ran the
paired cloudy challenge once. No threshold, partition, noise scale, seed or gate
was changed, and the run was not retried.

Validation accepted 42 of 48 states (87.5%) and covered 36 of those 42 accepted
states (85.7%). Those point estimates exceeded the frozen 70% acceptance and 85%
accepted-coverage floors. The corresponding Wilson 95% intervals were 75.3--94.1%
and 72.2--93.3%, which show the limited precision of this 48-state qualification.

The cloudy challenge accepted 77 of 240 states (32.1%; Wilson 95% 26.5--38.2%)
and covered 52 of the 77 accepted states (67.5%; Wilson 95% 56.5--76.9%). It
therefore failed both frozen gates. Every challenge interval passed the 300 K width
cap, so the failures came from the calibrated score and conditional coverage rather
than excessive interval width.

This result applies only to the Sonora Diamondback synthetic 16-band photometry,
the frozen gravity/metallicity split, the cloud-free support family and the fixed
0.05-mag Gaussian noise realization. It is not evidence about resolved spectra,
continuous temperature precision, real planets or other cloud models.

The machine-readable [record](record.json) binds the result to protocol commit
`8be8d61`, pre-outcome harness commit `46230b2`, the canonical grid, the exact
harness source and the full-grid noise realization.
