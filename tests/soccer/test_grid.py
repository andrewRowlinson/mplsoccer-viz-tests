"""Grids of pitches via draw and grid.
A marker is plotted on one known axis to check the axes ordering."""
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec

MPL = mpl_kwargs('grid')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_grid_horizontal(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, **kwargs)
    fig, ax = pitch.grid(nrows=3, ncols=4, figheight=5)
    pitch.scatter(pitch.dim.center_length, pitch.dim.center_width,
                  marker='x', s=500, ax=ax['pitch'][2, 1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw_grid_horizontal(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, **kwargs)
    fig, ax = pitch.draw(nrows=3, ncols=4, figsize=(6, 3.5))
    pitch.scatter(pitch.dim.center_length, pitch.dim.center_width,
                  marker='x', s=500, ax=ax[2, 1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_grid_vertical(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = VerticalPitch(pitch_type=pitch_type, **kwargs)
    fig, ax = pitch.grid(nrows=4, ncols=3, figheight=5)
    pitch.scatter(pitch.dim.center_length, pitch.dim.center_width,
                  marker='x', s=500, ax=ax['pitch'][2, 1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_draw_grid_vertical(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = VerticalPitch(pitch_type=pitch_type, **kwargs)
    fig, ax = pitch.draw(nrows=4, ncols=3, figsize=(3.5, 6))
    pitch.scatter(pitch.dim.center_length, pitch.dim.center_width,
                  marker='x', s=500, ax=ax[2, 1])
    return fig


def test_grid_dimensions():
    """Snapshot of the grid_dimensions helper output."""
    pitch = Pitch()
    grid_width, grid_height = pitch.grid_dimensions(figwidth=16, figheight=9,
                                                    nrows=2, ncols=3,
                                                    max_grid=1, space=0.05)
    assert grid_width == pytest.approx(1)
    assert grid_height == pytest.approx(0.81822413)
