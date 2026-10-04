"""Drawing the pitch for every pitch type: full, half, and with
random padding / stripes / colors."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, scale_pad

MPL = mpl_kwargs('pitch_draw')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, label=True, axis=True, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw_half(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True, half=True, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, label=True, axis=True,
                                   half=True, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw_pad_stripe(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    padding = rng.uniform(low=-15, high=15, size=4)
    pad_left, pad_right, pad_bottom, pad_top = [scale_pad(pad, name).round(1)
                                                for pad in padding]
    half = bool(rng.choice([True, False]))
    stripe = bool(rng.choice([True, False]))
    pitch_color = rng.choice(['#a4ed8e', 'grass'])
    # the 'grass' texture is drawn from numpy's global random state
    # (np.random.normal in mplsoccer), so seed it for a deterministic image
    np.random.seed(42)
    pitch = Pitch(pitch_type=pitch_type, label=True, axis=True,
                  half=half, stripe=stripe, pitch_color=pitch_color,
                  pad_left=pad_left, pad_right=pad_right,
                  pad_bottom=pad_bottom, pad_top=pad_top, **kwargs)
    # swap padding when vertical
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, label=True, axis=True,
                                   half=half, stripe=stripe, pitch_color=pitch_color,
                                   pad_bottom=pad_left, pad_top=pad_right,
                                   pad_left=pad_top, pad_right=pad_bottom, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])
    return fig
