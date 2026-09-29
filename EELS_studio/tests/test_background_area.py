"""Integrate the selected physical energy interval, including fractional bins."""
import numpy as np
import pytest

from background_area import integrate_area, symmetric_bounds, normalize_to_original


def test_linear_curve_on_irregular_grid_with_interpolated_endpoints():
    energy = np.array([0., 1., 2.5, 4.])
    intensity = 2 * energy + 1
    assert integrate_area(energy, intensity, .5, 3.5) == pytest.approx(15.)


def test_interval_inside_a_single_bin():
    assert integrate_area([0., 1., 2.], [2., 2., 2.], .2, .7) == pytest.approx(1.)


def test_negative_corrected_intensity_is_retained():
    assert integrate_area([0., 1., 2., 3., 4.], [-2., -1., 0., 1., 2.], 0., 3.) == pytest.approx(-1.5)


def test_masked_samples_outside_the_interval_are_allowed():
    assert integrate_area([0., 1., 2., 3.], [np.nan, 2., 4., np.nan], 1., 2.) == pytest.approx(3.)


@pytest.mark.parametrize('low, high', [(0., 2.), (1., 2.5), (1.1, 2.9)])
def test_missing_samples_cannot_be_bridged(low, high):
    with pytest.raises(ValueError, match='valid samples'):
        integrate_area([0., 1., 2., 3.], [1., np.nan, 3., 4.], low, high)


@pytest.mark.parametrize('low, high', [(2., 1.), (1., 1.), (-1., 1.), (1., 4.), (np.nan, 2.)])
def test_invalid_or_out_of_coverage_intervals_are_rejected(low, high):
    with pytest.raises(ValueError):
        integrate_area([0., 1., 2., 3.], [1., 1., 1., 1.], low, high)


def test_symmetric_bounds_use_width_on_each_side_of_center():
    assert symmetric_bounds(60., 10.) == (50., 70.)
    assert symmetric_bounds(-5., 2.) == (-7., -3.)


@pytest.mark.parametrize('center, width', [(0., 0.), (1., -2.), (np.nan, 2.), (1., np.inf)])
def test_invalid_center_or_half_width_is_rejected(center, width):
    with pytest.raises(ValueError):
        symmetric_bounds(center, width)


def test_normalization_uses_whole_original_spectrum_including_negative_energy():
    total, fraction = normalize_to_original(1., [-2., -1., 0., 1., 2.], [2., 2., 2., 2., 2.])
    assert total == pytest.approx(8.)
    assert fraction == pytest.approx(.125)


def test_normalization_retains_signed_corrected_area():
    total, fraction = normalize_to_original(-.5, [-2., 0., 2.], [2., 2., 2.])
    assert total == pytest.approx(8.)
    assert fraction == pytest.approx(-.0625)


def test_normalization_is_invariant_to_common_intensity_scaling():
    _, fraction = normalize_to_original(10., [-2., 0., 2.], [20., 20., 20.])
    assert fraction == pytest.approx(.125)


@pytest.mark.parametrize('intensity', [[0., 0.], [-1., 1.], [-1., 1. + 1e-14]])
def test_zero_or_cancelling_total_has_no_normalized_value(intensity):
    total, fraction = normalize_to_original(1., [0., 1.], intensity)
    assert np.isfinite(total)
    assert fraction is None
