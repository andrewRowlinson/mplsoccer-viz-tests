"""Arrows and comet lines on the pitch."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('arrows_lines')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_arrows_and_lines(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5, 5))

    pitch.draw(ax[0, 0])
    pitch.draw(ax[1, 1])
    pitch_vertical.draw(ax[0, 1])
    pitch_vertical.draw(ax[1, 0])

    x, y = random_points(pitch.dim, 5, rng)
    x_end, y_end = random_points(pitch.dim, 5, rng)

    pitch.arrows(x, y, x_end, y_end, ax=ax[0, 0], color='red')
    pitch_vertical.arrows(x, y, x_end, y_end, ax=ax[0, 1], color='red')
    pitch_vertical.lines(x, y, x_end, y_end, ax=ax[1, 0], comet=True,
                         transparent=True, color='red')
    pitch.lines(x, y, x_end, y_end, ax=ax[1, 1], comet=True,
                transparent=True, color='red')
    return fig
