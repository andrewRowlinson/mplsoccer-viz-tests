"""Flow maps and rotated markers.

Each pitch uses its own bin statistic so the vertical flow is
tested properly."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch, arrowhead_marker

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points, size_varies

MPL = mpl_kwargs('flow_and_angles')


def flow_figure(name, arrow_type):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])

    x, y = random_points(pitch.dim, 5, rng)
    x_end, y_end = random_points(pitch.dim, 5, rng)

    pitch.arrows(x, y, x_end, y_end, ax=ax[0], color='red')
    pitch.flow(x, y, x_end, y_end, arrow_type=arrow_type, ax=ax[0])
    stats = pitch.bin_statistic(x, y)
    pitch.heatmap(stats, ax=ax[0], alpha=0.1)

    pitch_vertical.arrows(x, y, x_end, y_end, ax=ax[1], color='red')
    pitch_vertical.flow(x, y, x_end, y_end, arrow_type=arrow_type, ax=ax[1])
    stats_vertical = pitch_vertical.bin_statistic(x, y)
    pitch_vertical.heatmap(stats_vertical, ax=ax[1], alpha=0.1)
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_flow(name):
    return flow_figure(name, arrow_type='same')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_flow_average(name):
    return flow_figure(name, arrow_type='average')


def rotated_markers_figure(pitch_type, kwargs):
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])

    x, y = random_points(pitch.dim, 5, rng)
    x_end, y_end = random_points(pitch.dim, 5, rng)

    pitch.arrows(x, y, x_end, y_end, ax=ax[0], alpha=0.2, color='red')
    pitch_vertical.arrows(x, y, x_end, y_end, ax=ax[1], alpha=0.2, color='red')

    angle, distance = pitch.calculate_angle_and_distance(x, y, x_end, y_end,
                                                         degrees=True)
    pitch.scatter(x, y, rotation_degrees=angle, s=100,
                  marker=arrowhead_marker, ax=ax[0])

    angle_vertical, distance_vertical = pitch_vertical.calculate_angle_and_distance(
        x, y, x_end, y_end, degrees=True)
    pitch_vertical.scatter(x, y, rotation_degrees=angle_vertical, s=100,
                           marker=arrowhead_marker, ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_rotated_markers(name):
    pitch_type, kwargs = pitch_spec(name)
    return rotated_markers_figure(pitch_type, kwargs)


@pytest.mark.parametrize('name', size_varies)
@pytest.mark.mpl_image_compare(**MPL)
def test_rotated_markers_wide_pitch(name):
    """A pitch wider (117) than it is long (87): the marker rotations must
    still line up with the arrows under an unusual aspect ratio. Only the
    size-varying pitch types accept custom dimensions."""
    return rotated_markers_figure(name, {'pitch_length': 87, 'pitch_width': 117})
