"""Positive and negative padding on each side of the pitch.
Each figure shows all eight
pad settings (top/bottom/left/right, +20/-20) for one pitch type."""
import matplotlib.pyplot as plt
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, scale_pad

MPL = mpl_kwargs('pad')

PAD_SETTINGS = [{'pad_top': 20}, {'pad_top': -20},
                {'pad_bottom': 20}, {'pad_bottom': -20},
                {'pad_left': 20}, {'pad_left': -20},
                {'pad_right': 20}, {'pad_right': -20}]


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_horizontal(name):
    pitch_type, kwargs = pitch_spec(name)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
    for pad_kwargs, ax in zip(PAD_SETTINGS, axs.flat):
        pad_kwargs = {key: scale_pad(pad, name) for key, pad in pad_kwargs.items()}
        pitch = Pitch(pitch_type=pitch_type, label=True, axis=True,
                      **pad_kwargs, **kwargs)
        pitch.draw(ax=ax)
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pad_vertical(name):
    pitch_type, kwargs = pitch_spec(name)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(8, 5))
    for pad_kwargs, ax in zip(PAD_SETTINGS, axs.flat):
        pad_kwargs = {key: scale_pad(pad, name) for key, pad in pad_kwargs.items()}
        pitch = VerticalPitch(pitch_type=pitch_type, label=True, axis=True,
                              **pad_kwargs, **kwargs)
        pitch.draw(ax=ax)
    return fig
