"""Standardizer coordinate transforms.

The round-trip and chain tests are numeric; the scatter test visually
checks points keep their relative positions after transforming."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, Standardizer

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('standardizer')

SIZE_KWARGS = {'width_from': 68, 'length_from': 105,
               'width_to': 68, 'length_to': 105}


@pytest.mark.parametrize('name_to', PITCH_TYPES)
@pytest.mark.parametrize('name_from', PITCH_TYPES)
def test_standardizer_round_trip(name_from, name_to):
    pitch_type_from, _ = pitch_spec(name_from)
    pitch_type_to, _ = pitch_spec(name_to)
    rng = np.random.default_rng(42)
    standard = Standardizer(pitch_from=pitch_type_from, pitch_to=pitch_type_to,
                            **SIZE_KWARGS)
    x = rng.uniform(low=standard.dim_from.pitch_extent[0],
                    high=standard.dim_from.pitch_extent[1], size=10_000)
    y = rng.uniform(low=standard.dim_from.pitch_extent[2],
                    high=standard.dim_from.pitch_extent[3], size=10_000)
    x_std, y_std = standard.transform(x, y)
    x_reverse, y_reverse = standard.transform(x_std, y_std, reverse=True)
    assert np.abs(x - x_reverse).sum() < 1e-7
    assert np.abs(y - y_reverse).sum() < 1e-7


def test_standardizer_chain():
    """Transform through every pitch type and back to the start."""
    rng = np.random.default_rng(42)
    pitch_types = [pitch_spec(name)[0] for name in PITCH_TYPES]
    pitch_types_shift = pitch_types[1:] + pitch_types[:1]
    dim_start = Standardizer(pitch_from=pitch_types[0], pitch_to=pitch_types[0],
                             **SIZE_KWARGS).dim_from
    x = rng.uniform(low=dim_start.pitch_extent[0],
                    high=dim_start.pitch_extent[1], size=10_000)
    y = rng.uniform(low=dim_start.pitch_extent[2],
                    high=dim_start.pitch_extent[3], size=10_000)
    x_copy = x.copy()
    y_copy = y.copy()
    for pitch_from, pitch_to in zip(pitch_types, pitch_types_shift):
        standard = Standardizer(pitch_from=pitch_from, pitch_to=pitch_to,
                                **SIZE_KWARGS)
        x, y = standard.transform(x, y)
    assert np.abs(x - x_copy).sum() < 1e-6
    assert np.abs(y - y_copy).sum() < 1e-6


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_standardizer_scatter(name):
    """Transform points from statsbomb to each pitch type and plot both."""
    pitch_type_to, kwargs_to = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch_from = Pitch(pitch_type='statsbomb')
    pitch_to = Pitch(pitch_type=pitch_type_to, **kwargs_to)
    x, y = random_points(pitch_from.dim, 100, rng)
    standard = Standardizer(pitch_from='statsbomb', pitch_to=pitch_type_to,
                            **SIZE_KWARGS)
    x_std, y_std = standard.transform(x, y)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 2.5))
    pitch_from.draw(ax=ax[0])
    pitch_to.draw(ax=ax[1])
    pitch_from.scatter(x, y, ax=ax[0])
    pitch_to.scatter(x_std, y_std, ax=ax[1])
    return fig
