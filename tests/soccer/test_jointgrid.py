"""Jointgrid with seaborn marginal kde plots."""
import numpy as np
import pytest
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('jointgrid')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_jointgrid_horizontal(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    x, y = random_points(pitch.dim, 300, rng)

    fig, ax = pitch.jointgrid(figheight=5, ax_bottom=True, space=0.01,
                              bottom=0.11, title_height=0.02)
    pitch.hexbin(x, y, ax=ax['pitch'])
    sns.kdeplot(y=y, fill=True, ax=ax['left'])
    sns.kdeplot(y=y, fill=True, ax=ax['right'])
    sns.kdeplot(x=x, fill=True, ax=ax['top'])
    sns.kdeplot(x=x, fill=True, ax=ax['bottom'])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_jointgrid_vertical(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    x, y = random_points(pitch.dim, 300, rng)

    fig, ax = pitch.jointgrid(figheight=5, ax_bottom=True, space=0.01)
    pitch.hexbin(x, y, ax=ax['pitch'])
    # the data axes swap on a vertical pitch
    sns.kdeplot(y=x, fill=True, ax=ax['left'])
    sns.kdeplot(y=x, fill=True, ax=ax['right'])
    sns.kdeplot(x=y, fill=True, ax=ax['top'])
    sns.kdeplot(x=y, fill=True, ax=ax['bottom'])
    return fig
