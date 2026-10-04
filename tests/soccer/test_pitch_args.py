"""Visual Pitch constructor arguments not covered by the other test modules:
goal types, spot types, positional lines / middle shading / corner arcs,
and the cosmetic line styling arguments."""
import matplotlib.pyplot as plt
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec

MPL = mpl_kwargs('pitch_args')

GOAL_TYPES = ['line', 'box', 'circle']


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_goal_types(name):
    pitch_type, kwargs = pitch_spec(name)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))
    for col, goal_type in enumerate(GOAL_TYPES):
        pitch = Pitch(pitch_type=pitch_type, goal_type=goal_type, **kwargs)
        pitch_vertical = VerticalPitch(pitch_type=pitch_type,
                                       goal_type=goal_type, **kwargs)
        pitch.draw(ax=axs[0, col])
        pitch_vertical.draw(ax=axs[1, col])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_spot_types(name):
    pitch_type, kwargs = pitch_spec(name)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))
    for col, spot_type in enumerate(['circle', 'square']):
        # a larger spot_scale than the default so differences are visible
        pitch = Pitch(pitch_type=pitch_type, spot_type=spot_type,
                      spot_scale=0.05, **kwargs)
        pitch_vertical = VerticalPitch(pitch_type=pitch_type, spot_type=spot_type,
                                       spot_scale=0.05, **kwargs)
        pitch.draw(ax=axs[0, col])
        pitch_vertical.draw(ax=axs[1, col])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pitch_markings(name):
    """positional (Juego de Posicion) lines, shaded middle third and
    corner arcs, each drawn from the pitch-type dimensions."""
    pitch_type, kwargs = pitch_spec(name)
    marking_kwargs = [{'positional': True}, {'shade_middle': True},
                      {'corner_arcs': True}]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 5))
    for col, marking in enumerate(marking_kwargs):
        pitch = Pitch(pitch_type=pitch_type, **marking, **kwargs)
        pitch_vertical = VerticalPitch(pitch_type=pitch_type, **marking, **kwargs)
        pitch.draw(ax=axs[0, col])
        pitch_vertical.draw(ax=axs[1, col])
    return fig


@pytest.mark.mpl_image_compare(**MPL)
def test_line_styling():
    """Cosmetic styling arguments on the default pitch type."""
    styles = [
        {'line_color': 'red', 'linewidth': 4, 'linestyle': '--',
         'line_alpha': 0.7},
        {'goal_type': 'box', 'goal_linestyle': ':', 'goal_alpha': 0.5,
         'linewidth': 1},
        {'stripe': True, 'stripe_color': '#a0c8f0', 'pitch_color': '#f0f8ff'},
        {'positional': True, 'positional_color': 'blue',
         'positional_linestyle': '--', 'positional_linewidth': 0.5,
         'shade_middle': True, 'shade_color': '#fffacd'},
    ]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))
    for style, ax in zip(styles, axs.flat):
        pitch = Pitch(**style)
        pitch.draw(ax=ax)
    return fig
