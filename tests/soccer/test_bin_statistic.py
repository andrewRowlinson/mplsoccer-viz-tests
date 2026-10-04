"""bin_statistic and bin_statistic_positional count all points, including
points exactly on the bin edges and pitch sides."""
import numpy as np
import pytest
from mplsoccer import Pitch

from helpers import PITCH_TYPES, pitch_spec, random_points

NUM_POINTS = 200_000


@pytest.mark.parametrize('name', PITCH_TYPES)
def test_bin_statistic_counts_all_points(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    x, y = random_points(pitch.dim, NUM_POINTS, rng)
    stats = pitch.bin_statistic(x, y)
    assert stats['statistic'].sum() == NUM_POINTS


@pytest.mark.parametrize('name', PITCH_TYPES)
def test_bin_statistic_positional_counts_all_points(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    x, y = random_points(pitch.dim, NUM_POINTS, rng)
    stats = pitch.bin_statistic_positional(x, y)
    assert stats['statistic'].sum() == NUM_POINTS


@pytest.mark.parametrize('name', PITCH_TYPES)
def test_bin_statistic_positional_y_markings(name):
    """Points exactly on the horizontal (y) bin edges are counted once."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    y = np.tile(pitch.dim.y_markings_sorted, 10_000)
    x = rng.uniform(low=pitch.dim.pitch_extent[0],
                    high=pitch.dim.pitch_extent[1], size=y.size)
    stats = pitch.bin_statistic_positional(x, y)
    assert stats['statistic'].sum() == y.size


@pytest.mark.parametrize('name', PITCH_TYPES)
def test_bin_statistic_positional_x_markings(name):
    """Points exactly on the vertical (x) bin edges are counted once."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    x = np.tile(pitch.dim.x_markings_sorted, 10_000)
    y = rng.uniform(low=pitch.dim.pitch_extent[2],
                    high=pitch.dim.pitch_extent[3], size=x.size)
    stats = pitch.bin_statistic_positional(x, y)
    assert stats['statistic'].sum() == x.size
