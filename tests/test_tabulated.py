import numpy as np
import pytest

from tierra_plus.forward.tabulated import bin_tabulated_spectrum
from tierra_plus.models import Spectrum


def test_piecewise_linear_bin_integrals_and_units():
    model = Spectrum([1.0, 2.0, 3.0], [1.0, 3.0, 1.0])
    result = bin_tabulated_spectrum(model, [1.0, 1.5, 2.5, 3.0])
    np.testing.assert_allclose(result.flux, [1.5, 2.5, 1.5])
    np.testing.assert_allclose(result.wavelength_microns, [1.25, 2.0, 2.75])


@pytest.mark.parametrize("edges", [[0, 2], [1, 4], [1, 1, 3], [1, np.nan], [[1, 3]]])
def test_invalid_bins_fail_without_extrapolation(edges):
    with pytest.raises(ValueError):
        bin_tabulated_spectrum(Spectrum([1, 2, 3], [1, 2, 3]), edges)


def test_uncertainty_propagation_is_not_silently_discarded():
    with pytest.raises(ValueError, match="uncertainty"):
        bin_tabulated_spectrum(Spectrum([1, 2], [1, 2], [0.1, 0.1]), [1, 2])
