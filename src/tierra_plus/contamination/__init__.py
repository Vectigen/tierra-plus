"""Transit light-source contamination from supplied stellar intensity spectra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tierra_plus.models import Spectrum


def contaminate_transit(
    planet: Spectrum,
    photosphere: Sequence[float],
    spots: Sequence[float],
    faculae: Sequence[float],
    spot_fraction: float = 0.0,
    facula_fraction: float = 0.0,
) -> Spectrum:
    """Apply unocculted surface heterogeneity to a quiet-chord transit depth.

    Parameters
    ----------
    planet
        Planet-only transit depths on a common wavelength grid.
    photosphere, spots, faculae
        Positive specific intensities on ``planet``'s grid in the same units.
    spot_fraction, facula_fraction
        Disk covering fractions, non-negative with a sum at most one.

    Notes
    -----
    Depth is multiplied by I_phot / [(1-f_s-f_f) I_phot + f_s I_spot + f_f I_fac].
    This assumes a quiet transit chord and ignores limb dependence and occulted
    active regions. See Rackham et al. (2018), doi:10.3847/1538-4357/aaa08c.
    Stellar atmosphere intensities must be supplied; no blackbody substitute is used.
    """
    fractions = np.asarray([spot_fraction, facula_fraction])
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or fractions.sum() > 1:
        raise ValueError("Covering fractions must be finite, nonnegative and sum to at most one")
    intensities = np.asarray([photosphere, spots, faculae], dtype=float)
    if intensities.shape != (3, len(planet.flux)):
        raise ValueError("All intensity spectra must match the planet wavelength grid")
    if not np.all(np.isfinite(intensities)) or np.any(intensities <= 0):
        raise ValueError("Stellar intensities must be finite and positive")
    quiet, spot, facula = intensities
    disk = (1 - fractions.sum()) * quiet + spot_fraction * spot + facula_fraction * facula
    factor = quiet / disk
    sigma = None if planet.uncertainty is None else tuple(np.asarray(planet.uncertainty) * factor)
    return Spectrum(planet.wavelength_microns, tuple(np.asarray(planet.flux) * factor), sigma)
