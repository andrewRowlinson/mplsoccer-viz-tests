"""Surface plots on a heavily padded pitch: heatmaps must tile exactly the
pitch area, and hexbin/kdeplot are clipped to the pitch boundary by
mplsoccer, so nothing should bleed into the padding. Points are placed at
the pitch corners on purpose so any overhang would be visible.

Note: in the hexbin test the horizontal and vertical panels show different
tessellations - that is expected (see test_hexbin.py); the thing under
test here is only that no hexagon crosses the pitch boundary."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points, scale_pad

MPL = mpl_kwargs('pad_clip')

PAD = 15


def padded_pitches(name):
    pitch_type, kwargs = pitch_spec(name)
    pad = scale_pad(PAD, name)
    pad_kwargs = {'pad_left': pad, 'pad_right': pad,
                  'pad_bottom': pad, 'pad_top': pad}
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **pad_kwargs, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2,
                                   **pad_kwargs, **kwargs)
    return pitch, pitch_vertical


def points_with_corners(dim, size, rng):
    """Random points plus the four pitch corners to stress the boundary."""
    x, y = random_points(dim, size, rng)
    corners_x = np.array([dim.pitch_extent[0], dim.pitch_extent[0],
                          dim.pitch_extent[1], dim.pitch_extent[1]])
    corners_y = np.array([dim.pitch_extent[2], dim.pitch_extent[3],
                          dim.pitch_extent[2], dim.pitch_extent[3]])
    return np.append(x, corners_x), np.append(y, corners_y)


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_clip_heatmap(name):
    pitch, pitch_vertical = padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = points_with_corners(pitch.dim, 20, rng)

    pitch.draw(ax[0])
    stats = pitch.bin_statistic(x, y)
    pitch.heatmap(stats, ax=ax[0])

    pitch_vertical.draw(ax[1])
    stats_vertical = pitch_vertical.bin_statistic(x, y)
    pitch_vertical.heatmap(stats_vertical, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_clip_heatmap_positional(name):
    pitch, pitch_vertical = padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = points_with_corners(pitch.dim, 20, rng)

    pitch.draw(ax[0])
    stats = pitch.bin_statistic_positional(x, y)
    pitch.heatmap_positional(stats, ax=ax[0])

    pitch_vertical.draw(ax[1])
    stats_vertical = pitch_vertical.bin_statistic_positional(x, y)
    pitch_vertical.heatmap_positional(stats_vertical, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_clip_hexbin(name):
    pitch, pitch_vertical = padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = points_with_corners(pitch.dim, 40, rng)

    pitch.draw(ax[0])
    pitch.hexbin(x, y, gridsize=8, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.hexbin(x, y, gridsize=8, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_clip_kdeplot(name):
    pitch, pitch_vertical = padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    # cluster the points near one corner so the kde spills over two edges
    x, y = random_points(pitch.dim, 60, rng)
    x = pitch.dim.pitch_extent[0] + (x - pitch.dim.pitch_extent[0]) / 3
    y = pitch.dim.pitch_extent[2] + (y - pitch.dim.pitch_extent[2]) / 3

    pitch.draw(ax[0])
    pitch.kdeplot(x, y, fill=True, levels=25, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.kdeplot(x, y, fill=True, levels=25, ax=ax[1])
    return fig


# ---- negative padding: the view is smaller than the pitch, and nothing
# should render beyond the shrunken axes into the figure ----

NEGATIVE_PAD = -15


def negative_padded_pitches(name):
    pitch_type, kwargs = pitch_spec(name)
    pad = scale_pad(NEGATIVE_PAD, name)
    pad_kwargs = {'pad_left': pad, 'pad_right': pad,
                  'pad_bottom': pad, 'pad_top': pad}
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **pad_kwargs, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2,
                                   **pad_kwargs, **kwargs)
    return pitch, pitch_vertical


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_negative_pad_heatmap(name):
    pitch, pitch_vertical = negative_padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 40, rng)

    pitch.draw(ax[0])
    stats = pitch.bin_statistic(x, y)
    pitch.heatmap(stats, ax=ax[0])

    pitch_vertical.draw(ax[1])
    stats_vertical = pitch_vertical.bin_statistic(x, y)
    pitch_vertical.heatmap(stats_vertical, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_negative_pad_hexbin(name):
    pitch, pitch_vertical = negative_padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 40, rng)

    pitch.draw(ax[0])
    pitch.hexbin(x, y, gridsize=8, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.hexbin(x, y, gridsize=8, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_negative_pad_kdeplot(name):
    pitch, pitch_vertical = negative_padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 60, rng)

    pitch.draw(ax[0])
    pitch.kdeplot(x, y, fill=True, levels=25, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.kdeplot(x, y, fill=True, levels=25, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_negative_pad_sonar_grid(name):
    """Since mplsoccer 1.8, sonar_grid skips inset axes whose bin center
    falls outside the shrunken view (exclude_outside=True by default), so
    only sonars for the visible part of the pitch are drawn. Before 1.8
    the out-of-view sonars leaked outside the axes into the figure."""
    pitch, pitch_vertical = negative_padded_pitches(name)
    pitch_type = pitch_spec(name)[0]
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 40, rng)
    x_end, y_end = random_points(pitch.dim, 40, rng)
    width = pitch.dim.width / 4 if isinstance(pitch_type, str) else 10

    pitch.draw(ax[0])
    angle, distance = pitch.calculate_angle_and_distance(x, y, x_end, y_end)
    bs_sonar = pitch.bin_statistic_sonar(x, y, angle, bins=(5, 4, 10))
    pitch.sonar_grid(bs_sonar, width=width, ax=ax[0])

    pitch_vertical.draw(ax[1])
    angle_vertical, distance_vertical = pitch_vertical.calculate_angle_and_distance(
        x, y, x_end, y_end)
    bs_sonar_vertical = pitch_vertical.bin_statistic_sonar(x, y, angle_vertical,
                                                           bins=(5, 4, 10))
    pitch_vertical.sonar_grid(bs_sonar_vertical, width=width, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
def test_negative_pad_label_heatmap_clipped(name):
    """Since mplsoccer 1.8, label_heatmap draws its texts with clip_on=True,
    so labels for bins outside a negative-padded view are created but not
    rendered (pass clip_on=False for the pre-1.8 leaking behaviour)."""
    pitch, _ = negative_padded_pitches(name)
    rng = np.random.default_rng(42)
    fig, ax = pitch.draw(figsize=(4, 3))
    x, y = random_points(pitch.dim, 40, rng)
    stats = pitch.bin_statistic(x, y)
    pitch.heatmap(stats, ax=ax)
    texts = pitch.label_heatmap(stats, color='white', ax=ax)
    assert texts, 'expected some heatmap labels'
    assert all(text.get_clip_on() for text in texts)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    outside = [text for text in texts
               if not (min(xlim) <= text.get_position()[0] <= max(xlim)
                       and min(ylim) <= text.get_position()[1] <= max(ylim))]
    assert outside, 'expected labels positioned outside the view (clipped)'
