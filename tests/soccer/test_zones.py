"""Custom zone layouts (mplsoccer 1.8 zones API).

Covers positional_zones, mirror_zones, draw_zones, bin_statistic_zones,
heatmap_zones and bin_statistic_sonar_zones/sonar_zones.

Note: remove_text only strips titles and tick labels, so the label_heatmap
and draw_zones texts ARE part of the baseline images (matplotlib wheels
bundle freetype, so the glyphs are stable for pip installs). statsbomb and
wyscout baselines legitimately differ from the other pitch types: their
coordinate systems place the pitch landmarks (and so the landmark-derived
zones) at different fractional positions, and points near the zone edges
bin differently."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('zones')


def half_pitch_zones(dim):
    """A custom layout for the right half of the pitch built from the pitch
    landmarks (from the plot_heatmap_zones example). The y pairs are sorted
    so y0 < y1 on both regular and inverted-y pitch types."""
    def rect(x0, x1, ylo, yhi):
        y0, y1 = sorted((ylo, yhi))
        return (x0, x1, y0, y1)
    return [rect(dim.penalty_area_right, dim.right,
                 dim.top, dim.penalty_area_top),
            rect(dim.positional_x[-3], dim.penalty_area_right,
                 dim.top, dim.penalty_area_top),
            rect(dim.center_length, dim.positional_x[-3],
                 dim.top, dim.penalty_area_top),
            rect(dim.six_yard_right, dim.right,
                 dim.penalty_area_top, dim.six_yard_top),
            rect(dim.penalty_area_right, dim.six_yard_right,
                 dim.penalty_area_top, dim.six_yard_top),
            rect(dim.center_length, dim.penalty_area_right,
                 dim.penalty_area_top, dim.six_yard_top),
            rect(dim.six_yard_right, dim.right,
                 dim.six_yard_top, dim.six_yard_bottom),
            rect(dim.penalty_area_right, dim.six_yard_right,
                 dim.six_yard_top, dim.six_yard_bottom),
            rect(dim.center_length, dim.penalty_area_right,
                 dim.six_yard_top, dim.six_yard_bottom),
            ]


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw_zones_mirror(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    zones = half_pitch_zones(pitch.dim)
    names = [f'zone-{i}' for i in range(len(zones))]
    zones_full, names_full = pitch.mirror_zones(zones, names=names, axis='both')

    pitch.draw(ax[0])
    pitch.draw_zones(zones_full, names_full, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.draw_zones(zones_full, names_full, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap_zones(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 40, rng)
    zones, names = pitch.mirror_zones(half_pitch_zones(pitch.dim), axis='both')

    pitch.draw(ax[0])
    pitch.scatter(x, y, ax=ax[0], color='red', zorder=3, s=10)
    stats = pitch.bin_statistic_zones(x, y, zones, normalize=True)
    pitch.heatmap_zones(stats, ax=ax[0], cmap='viridis', edgecolors='black')
    pitch.label_heatmap(stats, color='white', ax=ax[0], ha='center', va='center',
                        str_format='{:.0%}')

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, ax=ax[1], color='red', zorder=3, s=10)
    stats_vertical = pitch_vertical.bin_statistic_zones(x, y, zones, normalize=True)
    pitch_vertical.heatmap_zones(stats_vertical, ax=ax[1], cmap='viridis',
                                 edgecolors='black')
    pitch_vertical.label_heatmap(stats_vertical, color='white', ax=ax[1],
                                 ha='center', va='center', str_format='{:.0%}')
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_heatmap_zones_positional_variants(name):
    """The 'horizontal' and 'vertical' positional_zones layouts plotted
    through bin_statistic_zones/heatmap_zones ('full' is already covered by
    the positional tests in test_heatmap.py)."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 7))
    x, y = random_points(pitch.dim, 40, rng)

    for row, positional in enumerate(['horizontal', 'vertical']):
        for col, p in enumerate([pitch, pitch_vertical]):
            ax = axs[row, col]
            p.draw(ax)
            p.scatter(x, y, ax=ax, color='red', zorder=3, s=10)
            zones, names = p.positional_zones(positional)
            stats = p.bin_statistic_zones(x, y, zones, names=names)
            p.heatmap_zones(stats, ax=ax, cmap='viridis', edgecolors='black')
            p.label_heatmap(stats, color='white', ax=ax, ha='center', va='center')
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_sonar_zones(name):
    """A few passes drawn as arrows with a sonar per zone, so the slices
    can be checked by eye against the pass directions."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 8, rng)
    x_end, y_end = random_points(pitch.dim, 8, rng)
    extent = pitch.dim.pitch_extent
    width = (extent[1] - extent[0]) / 9
    zones, names = pitch.positional_zones('full')

    for a, p in zip(ax, [pitch, pitch_vertical]):
        p.draw(a)
        p.arrows(x, y, x_end, y_end, ax=a, color='red')
        angle, distance = p.calculate_angle_and_distance(x, y, x_end, y_end)
        bs_count = p.bin_statistic_sonar_zones(x, y, angle, zones, names=names,
                                               angle_bins=8, center=True)
        zone_count = p.bin_statistic_zones(x, y, zones, names=names)
        p.heatmap_zones(zone_count, ax=a, cmap='Blues', edgecolors='black', alpha=0.3)
        p.sonar_zones(bs_count, fc='cornflowerblue', ec='black',
                      width=width, zorder=3, ax=a)
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_sonar_zones_stats_color(name):
    """Denser data exercising the stats_color/cmap path of sonar_zones."""
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 40, rng)
    x_end, y_end = random_points(pitch.dim, 40, rng)
    extent = pitch.dim.pitch_extent
    width = (extent[1] - extent[0]) / 9
    zones, names = pitch.positional_zones('full')

    for a, p in zip(ax, [pitch, pitch_vertical]):
        p.draw(a)
        angle, distance = p.calculate_angle_and_distance(x, y, x_end, y_end)
        bs_count = p.bin_statistic_sonar_zones(x, y, angle, zones, names=names,
                                               angle_bins=4, center=True)
        bs_mean = p.bin_statistic_sonar_zones(x, y, angle, zones, values=distance,
                                              statistic='mean', names=names,
                                              angle_bins=4, center=True)
        zone_count = p.bin_statistic_zones(x, y, zones, names=names)
        p.heatmap_zones(zone_count, ax=a, cmap='Blues', edgecolors='black', alpha=0.5)
        p.sonar_zones(bs_count, stats_color=bs_mean, cmap='viridis',
                      ec='black', width=width, zorder=3, ax=a)
    return fig
