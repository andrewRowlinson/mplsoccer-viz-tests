"""Heatmaps and positional heatmaps.

Note: remove_text only strips titles and tick labels, so the label_heatmap
texts ARE part of the baseline images. matplotlib wheels bundle their own
freetype, so the glyphs are stable across machines for pip installs."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import (mpl_kwargs, mirror_inverted_y, PITCH_TYPES, pitch_spec,
                     random_points)

MPL = mpl_kwargs('heatmap')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 5, rng)

    pitch.draw(ax[0])
    pitch.scatter(x, y, ax=ax[0], color='red', zorder=3)
    stats = pitch.bin_statistic(x, y)
    stats['statistic'][stats['statistic'] == 0] = np.nan
    pitch.heatmap(stats, ax=ax[0])
    pitch.label_heatmap(stats, color='white', ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, ax=ax[1], color='red', zorder=3)
    stats_vertical = pitch_vertical.bin_statistic(x, y)
    stats_vertical['statistic'][stats_vertical['statistic'] == 0] = np.nan
    pitch_vertical.heatmap(stats_vertical, ax=ax[1])
    pitch_vertical.label_heatmap(stats_vertical, color='white', ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap_positional(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 5, rng)

    pitch.draw(ax[0])
    pitch.scatter(x, y, ax=ax[0], color='red', zorder=3)
    stats = pitch.bin_statistic_positional(x, y)
    pitch.heatmap_positional(stats, ax=ax[0])
    pitch.label_heatmap(stats, color='white', ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, ax=ax[1], color='red', zorder=3)
    stats_vertical = pitch_vertical.bin_statistic_positional(x, y)
    pitch_vertical.heatmap_positional(stats_vertical, ax=ax[1])
    pitch_vertical.label_heatmap(stats_vertical, color='white', ax=ax[1])
    return fig


def positional_markings_figure(name, markings):
    """Points on the positional bin edges to check edge-inclusion behaviour."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, pitch_color='None',
                  axis=True, label=True, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2,
                                   pitch_color='None', axis=True, label=True,
                                   **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    if markings == 'x':
        x = pitch.dim.positional_x
        y = mirror_inverted_y(pitch.dim,
                              rng.uniform(low=pitch.dim.pitch_extent[2],
                                          high=pitch.dim.pitch_extent[3],
                                          size=x.size))
    else:
        y = pitch.dim.positional_y
        x = rng.uniform(low=pitch.dim.pitch_extent[0],
                        high=pitch.dim.pitch_extent[1], size=y.size)

    pitch.draw(ax[0])
    pitch.scatter(x, y, ax=ax[0], color='red', zorder=3)
    stats = pitch.bin_statistic_positional(x, y)
    pitch.heatmap_positional(stats, ax=ax[0], edgecolors='yellow')
    pitch.label_heatmap(stats, color='white', ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, ax=ax[1], color='red', zorder=3)
    stats_vertical = pitch_vertical.bin_statistic_positional(x, y)
    pitch_vertical.heatmap_positional(stats_vertical, ax=ax[1], edgecolors='yellow')
    pitch_vertical.label_heatmap(stats_vertical, color='white', ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap_positional_x_markings(name):
    return positional_markings_figure(name, markings='x')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap_positional_y_markings(name):
    return positional_markings_figure(name, markings='y')
