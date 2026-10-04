"""inset_image: an image embedded at pitch coordinates. An asymmetric
image is used so a flipped or rotated inset would fail the comparison.
(inset_axes itself is exercised by the sonar_grid and formation tests.)"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec

MPL = mpl_kwargs('insets')

# asymmetric in both axes: a gradient with a bright top-left corner block
IMAGE = np.arange(64, dtype=float).reshape(8, 8)
IMAGE[:3, :3] = 80


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_inset_image(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    height = pitch.dim.width / 4

    pitch.draw(ax[0])
    pitch.inset_image(pitch.dim.center_length, pitch.dim.center_width,
                      IMAGE, height=height, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.inset_image(pitch_vertical.dim.center_length,
                               pitch_vertical.dim.center_width,
                               IMAGE, height=height, ax=ax[1])
    return fig
