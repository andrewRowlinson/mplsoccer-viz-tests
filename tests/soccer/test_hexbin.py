"""Hexbin plots on the pitch.

Note: hexbin is the one mplsoccer method where the horizontal and vertical
orientations legitimately differ - the hexagon tessellation is computed in
display space, so rotating the pitch re-bins the points. Don't expect the
two panels to be rotated copies of each other."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('hexbin')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_hexbin(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 10, rng)

    pitch.draw(ax[0])
    pitch.scatter(x, y, ax=ax[0], color='red', zorder=3)
    pitch.hexbin(x, y, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, ax=ax[1], color='red', zorder=3)
    pitch_vertical.hexbin(x, y, ax=ax[1])
    return fig
