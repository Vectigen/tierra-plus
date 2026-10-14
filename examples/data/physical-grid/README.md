# Prepared PICASO atmosphere grids

These derived spectra are **CC-BY-4.0**, separately from the repository's code
license. Original creators: Sagnick Mukherjee, Sarah E. Moran, Kazumasa Ohno,
Natasha E. Batalha and Jonathan J. Fortney (2022).
Source: [PICASO 3.0 Atmospheric Models of WASP-39 b](https://zenodo.org/records/7236759),
DOI 10.5281/zenodo.7236759. Cite the papers requested by that record:
[Mukherjee et al.](https://arxiv.org/abs/2208.07836),
[Batalha et al.](https://arxiv.org/abs/1904.09355), and, for the photochemical
spectra, [Tsai et al.](https://arxiv.org/abs/2108.01790).

Modifications: sort wavelengths; integrate the supplied piecewise-linear transit
depth over the inherited PRISM midpoint bins with lower edge at least 2.3 µm;
repeat at four explicitly recorded wavelength displacements. No observed transit
depth was used. The included errors/geometry derive from the separately attributed
[public PRISM assets](../wasp39-prism/README.md).

Each directory contains a small `grid.npz` and `manifest.json`. The manifest
preserves archive size/MD5/SHA256, every original model-member hash, physical
coordinates, code attribution, split IDs, shift values and the prepared-array
checksum. The raw author ZIPs are not duplicated here. Regenerate from those ZIPs
with [the preparation script](../../prepare_physical_grid.py).

The equilibrium archive supplies 192 states; the photochemical archive supplies
116. Prepared-array axes are `(physical_state, wavelength_shift, spectral_bin)`.
Shift indices 0,1,2 correspond to `-halfwidth,0,+halfwidth`; index 3 is the separate
off-grid challenge at `+0.5*halfwidth`. The half-width is half the median retained
bin width. These are controlled stress values, not measured instrument errors.
