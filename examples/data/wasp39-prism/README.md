# WASP-39b published PRISM inputs

These two unmodified HDF5 members come from **Zafar Rustamkulov, David K. Sing,
Sagnick Mukherjee, Erin May, James Kirk, Everett Schlawin, Michael R. Line,
Caroline Piaulet, Aarynn Carter, Natasha E. Batalha and Jayesh Goyal (2022)**,
[*Products and Models for Early Release Science of the Exoplanet WASP-39b with
JWST NIRSpec PRISM*, Zenodo v1](https://doi.org/10.5281/zenodo.7388032).
The dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The repository's Apache license does not replace that data license.

| Archive member | Local file | SHA-256 |
| --- | --- | --- |
| `transit_spectra/FIREFLy_transit_spec.h5` | `FIREFLy_transit_spec.h5` | `02fa509fc8cd81eaa8451a378a72cb3df7488cfcf5ea83c3b74c978f978d884c` |
| `best_fit_model/best_fit_model.h5` | `best_fit_model.h5` | `108aa6aef6d3b21e23cd4f5b32f7150f420914b8267e03e5cb92d9c2c1d3b61c` |

The original archive is `W39_PRISM_data.zip` (284,909,288 bytes). Its Zenodo
advertised MD5 is recorded in `source.json`; the whole archive was **not** downloaded
or MD5-verified in this run. The selected members were obtained from HTTP byte
ranges, checked against ZIP member CRCs, and pinned by the SHA-256 values above.
Download the archive from the linked record to independently reproduce extraction.
The example checks both local SHA-256 values before opening either file.

The associated [Rustamkulov et al. (2023) paper](https://doi.org/10.1038/s41586-022-05677-y)
identifies FIREFLy as the baseline reduction and PICASO as its best-fit model.
Wavelengths are micrometres and transit depths are dimensionless `(Rp/Rstar)^2`;
uncertainties are published one-standard-deviation depth errors. The observed file
has 207 channels. The model has 147,878 points in descending wavelength order;
the example sorts this grid before integrating it.

The original authors selected the model using these data. Reusing it does not
provide an independent atmospheric inference or a fresh radiative-transfer run.
The example's wavelength cut, bin response and offset fit are described in
[the results note](../../results/wasp39-prism/README.md).
