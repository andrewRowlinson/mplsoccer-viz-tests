"""flip_side: mirroring coordinates into the other half of the pitch."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('flip_side')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_flip_side(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 8, rng)
    # keep the originals in the left half so the flip is easy to see
    x = pitch.dim.pitch_extent[0] + (x - pitch.dim.pitch_extent[0]) / 2
    flip = np.ones(x.size, dtype=bool)
    x_flip, y_flip = pitch.flip_side(x, y, flip)

    # flipping twice must return the original points
    x_back, y_back = pitch.flip_side(x_flip, y_flip, flip)
    np.testing.assert_allclose(x_back, x)
    np.testing.assert_allclose(y_back, y)

    pitch.draw(ax[0])
    pitch.scatter(x, y, color='red', ax=ax[0])
    pitch.scatter(x_flip, y_flip, color='blue', ax=ax[0])
    pitch.lines(x, y, x_flip, y_flip, color='grey', linewidth=1, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.scatter(x, y, color='red', ax=ax[1])
    pitch_vertical.scatter(x_flip, y_flip, color='blue', ax=ax[1])
    pitch_vertical.lines(x, y, x_flip, y_flip, color='grey', linewidth=1, ax=ax[1])
    return fig
