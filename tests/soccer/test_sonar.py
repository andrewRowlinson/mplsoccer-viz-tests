"""Sonar grids.

Each pitch uses its own statistics so the vertical sonar is tested
properly."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('sonar')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_sonar_grid(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])

    x, y = random_points(pitch.dim, 5, rng)
    x_end, y_end = random_points(pitch.dim, 5, rng)
    width = pitch.dim.width / 4 if isinstance(pitch_type, str) else 10

    pitch.arrows(x, y, x_end, y_end, ax=ax[0], color='red')
    stats = pitch.bin_statistic(x, y)
    pitch.heatmap(stats, ax=ax[0], alpha=0.1)
    angle, distance = pitch.calculate_angle_and_distance(x, y, x_end, y_end)
    bs_sonar = pitch.bin_statistic_sonar(x, y, angle, bins=(5, 4, 20))
    pitch.sonar_grid(bs_sonar, width=width, ax=ax[0])

    pitch_vertical.arrows(x, y, x_end, y_end, ax=ax[1], color='red')
    stats_vertical = pitch_vertical.bin_statistic(x, y)
    pitch_vertical.heatmap(stats_vertical, ax=ax[1], alpha=0.1)
    angle_vertical, distance_vertical = pitch_vertical.calculate_angle_and_distance(
        x, y, x_end, y_end)
    bs_sonar_vertical = pitch_vertical.bin_statistic_sonar(x, y, angle_vertical,
                                                           bins=(5, 4, 20))
    pitch_vertical.sonar_grid(bs_sonar_vertical, width=width, ax=ax[1])
    return fig
