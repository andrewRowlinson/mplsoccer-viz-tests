"""NaN handling on the default (statsbomb) pitch.

Where a figure has two pitches, the left pitch plots data containing NaNs
and the right pitch plots the same data with the NaNs dropped: both sides
should look identical."""
import numpy as np
import pytest
from mplsoccer import Pitch, arrowhead_marker

from helpers import mpl_kwargs

MPL = mpl_kwargs('nan')

X = [0, 3, 10, 20, 40, 60, 70, 80, 90, 100, 110, 120]
Y = [80, 20, 20, 30, 10, 80, 30, 20, 0, 40, 4, 0]
X_NAN = [0, 3, 10, 20, 40, np.nan, 70, 80, 90, 100, 110, 120]


def random_points_with_nan(rng, size=100, nan_probability=0.2):
    x = rng.uniform(0, 120, size=size)
    y = rng.uniform(0, 80, size=size)
    mask = rng.random(size) < nan_probability
    return x, y, mask


def random_lines_with_nan(rng, size=20, nan_probability=0.2):
    x = rng.uniform(0, 120, size=size)
    y = rng.uniform(0, 80, size=size)
    end_x = rng.uniform(0, 120, size=size)
    end_y = rng.uniform(0, 80, size=size)
    mask = rng.random(size) < nan_probability
    return x, y, end_x, end_y, mask


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_plot():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.plot(X, Y, color='red', marker='o', ax=ax)
    pitch.plot(X_NAN, Y, marker='x', markersize=20, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_scatter():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.scatter(X, Y, color='red', marker='o', ax=ax)
    pitch.scatter(X_NAN, Y, marker='x', s=200, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_scatter_football():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.scatter(X, Y, marker='x', s=200, ax=ax)
    pitch.scatter(X_NAN, Y, marker='football', s=200, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_scatter_rotation():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    rotation_degrees = np.linspace(0, 360, num=12)
    pitch.scatter(X, Y, rotation_degrees=rotation_degrees,
                  color='red', marker=arrowhead_marker, s=200, ax=ax)
    rotation_degrees = rotation_degrees.copy()
    rotation_degrees[[5, 6]] = np.nan
    pitch.scatter(X, Y, rotation_degrees=rotation_degrees,
                  color='blue', marker=arrowhead_marker, s=100, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_kdeplot():
    rng = np.random.default_rng(42)
    x, y, mask = random_points_with_nan(rng)
    x[mask] = np.nan
    y[mask] = np.nan
    pitch = Pitch()
    fig, axs = pitch.draw(ncols=2, figsize=(6, 2.5))
    pitch.kdeplot(x, y, ax=axs[0])
    pitch.kdeplot(x[~mask], y[~mask], ax=axs[1])
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_hexbin():
    rng = np.random.default_rng(42)
    x, y, mask = random_points_with_nan(rng)
    x[mask] = np.nan
    y[mask] = np.nan
    pitch = Pitch()
    fig, axs = pitch.draw(ncols=2, figsize=(6, 2.5))
    pitch.hexbin(x, y, ax=axs[0])
    pitch.hexbin(x[~mask], y[~mask], ax=axs[1])
    return fig


def test_nan_bin_statistic_raises():
    """NaN coordinates raise since scipy 1.18 (scipy/scipy#23751): NaN samples
    are rejected outright. Before that, scipy only checked the sample when
    bins was an int, and mplsoccer passes a tuple, so the NaN points were
    silently dropped. NaNs in *values* are still fine (see the mean tests)."""
    rng = np.random.default_rng(42)
    x, y, mask = random_points_with_nan(rng)
    x[mask] = np.nan
    y[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    with pytest.raises(ValueError, match='NaN'):
        pitch.bin_statistic(x, y)
    with pytest.raises(ValueError, match='NaN'):
        pitch.bin_statistic_positional(x, y)


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_heatmap_mean_statistic():
    rng = np.random.default_rng(42)
    x, y, mask = random_points_with_nan(rng)
    values = rng.uniform(0, 1, size=100)
    values[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, axs = pitch.draw(ncols=2, figsize=(6, 2.5))
    bs_nan = pitch.bin_statistic(x, y, values=values, statistic='mean')
    bs_no_nan = pitch.bin_statistic(x[~mask], y[~mask], values=values[~mask],
                                    statistic='mean')
    np.testing.assert_allclose(np.nansum(bs_nan['statistic']),
                               np.nansum(bs_no_nan['statistic']))
    pitch.heatmap(bs_nan, ax=axs[0])
    pitch.heatmap(bs_no_nan, ax=axs[1])
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_heatmap_positional_mean_statistic():
    rng = np.random.default_rng(42)
    x, y, mask = random_points_with_nan(rng)
    values = rng.uniform(0, 1, size=100)
    values[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, axs = pitch.draw(ncols=2, figsize=(6, 2.5))
    bs_nan = pitch.bin_statistic_positional(x, y, values=values, statistic='mean')
    bs_no_nan = pitch.bin_statistic_positional(x[~mask], y[~mask],
                                               values=values[~mask],
                                               statistic='mean')
    np.testing.assert_allclose(np.nansum(bs_nan['statistic']),
                               np.nansum(bs_no_nan['statistic']))
    pitch.heatmap_positional(bs_nan, ax=axs[0])
    pitch.heatmap_positional(bs_no_nan, ax=axs[1])
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_arrows():
    rng = np.random.default_rng(42)
    x, y, end_x, end_y, mask = random_lines_with_nan(rng)
    end_x[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.arrows(x, y, end_x, end_y, width=4, color='red', ax=ax)
    pitch.arrows(x[~mask], y[~mask], end_x[~mask], end_y[~mask], width=2, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_lines():
    rng = np.random.default_rng(42)
    x, y, end_x, end_y, mask = random_lines_with_nan(rng)
    end_x[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.lines(x, y, end_x, end_y, linewidth=4, color='red', ax=ax)
    pitch.lines(x[~mask], y[~mask], end_x[~mask], end_y[~mask], linewidth=2, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_lines_comet():
    rng = np.random.default_rng(42)
    x, y, end_x, end_y, mask = random_lines_with_nan(rng)
    end_x[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, ax = pitch.draw(figsize=(4, 3))
    pitch.lines(x, y, end_x, end_y, linewidth=4, color='red', comet=True, ax=ax)
    pitch.lines(x[~mask], y[~mask], end_x[~mask], end_y[~mask], linewidth=2,
                comet=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_nan_goal_angle():
    pitch = Pitch(line_zorder=2)
    fig, axs = pitch.draw(ncols=2, figsize=(6, 2.5))
    pitch.goal_angle([110, 95], [35, 45], alpha=0.5, ax=axs[0])
    pitch.goal_angle([110, np.nan], [35, 45], alpha=0.5, ax=axs[1])
    return fig


def test_nan_calculate_angle_and_distance():
    rng = np.random.default_rng(42)
    x, y, end_x, end_y, mask = random_lines_with_nan(rng)
    end_x[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    angle, distance = pitch.calculate_angle_and_distance(x, y, end_x, end_y)
    assert np.isnan(angle[mask]).all()
    assert np.isnan(distance[mask]).all()
    assert np.isfinite(angle[~mask]).all()
    assert np.isfinite(distance[~mask]).all()


def test_nan_flow_raises():
    """flow bins the start points, so NaN coordinates raise like bin_statistic
    (scipy >= 1.18); the NaN points were silently dropped before that."""
    rng = np.random.default_rng(42)
    x, y, end_x, end_y, mask = random_lines_with_nan(rng)
    x[mask] = np.nan
    pitch = Pitch(line_zorder=2)
    fig, ax = pitch.draw(figsize=(4, 3))
    with pytest.raises(ValueError, match='NaN'):
        pitch.flow(x, y, end_x, end_y, ax=ax)


def test_nan_voronoi_raises():
    """scipy's Voronoi (qhull) rejects NaN points, and has done for a long
    time (verified back to scipy 1.7), so voronoi raises rather than
    silently excluding the NaN player."""
    rng = np.random.default_rng(42)
    pitch = Pitch(line_zorder=2)
    x = rng.uniform(0, 120, size=22)
    y = rng.uniform(0, 80, size=22)
    teams = np.array([0] * 11 + [1] * 11)
    x[11] = np.nan
    with pytest.raises(ValueError):
        pitch.voronoi(x, y, teams)
