"""Bin a supplied tabulated spectrum with an explicit top-hat response."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tierra_plus.models import Spectrum


def bin_tabulated_spectrum(spectrum: Spectrum, edges: Sequence[float]) -> Spectrum:
    """Integrate piecewise-linear flux over contiguous bins, without extrapolation.

    This is a uniform wavelength-response approximation. It does not implement an
    instrument line-spread function or photon/throughput weighting. Input model
    uncertainties are rejected because propagation would require covariance.
    """
    bounds = np.asarray(edges, dtype=float)
    if (
        bounds.ndim != 1
        or len(bounds) < 2
        or not np.isfinite(bounds).all()
        or np.any(np.diff(bounds) <= 0)
    ):
        raise ValueError("Bin edges must be finite and strictly increasing")
    if spectrum.uncertainty is not None:
        raise ValueError("Tabulated model uncertainty propagation is not implemented")
    wave, flux = np.asarray(spectrum.wavelength_microns), np.asarray(spectrum.flux)
    if bounds[0] < wave[0] or bounds[-1] > wave[-1]:
        raise ValueError("Bin edges must lie inside the tabulated wavelength grid")
    values = []
    for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
        x = np.concatenate(([lower], wave[(wave > lower) & (wave < upper)], [upper]))
        y = np.interp(x, wave, flux)
        integral = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2)
        values.append(float(integral / (upper - lower)))
    return Spectrum(tuple((bounds[:-1] + bounds[1:]) / 2), tuple(values))
