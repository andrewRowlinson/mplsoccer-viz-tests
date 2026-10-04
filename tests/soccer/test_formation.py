"""Formation plotting and position lookups. The positions are defined per
pitch type from its dimensions, so the same formation should appear at the
same physical locations on every pitch type."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec

MPL = mpl_kwargs('formation')


# formation() does not support custom dimension objects (see
# test_formation_custom_dims_unsupported), so center_scale is excluded
@pytest.mark.parametrize('name', [name for name in PITCH_TYPES
                                  if name != 'center_scale'])
@pytest.mark.mpl_image_compare(**MPL)
def test_formation_scatter(name):
    pitch_type, kwargs = pitch_spec(name)
    # statsbomb position ids are ambiguous (e.g. RM is [12, 17]) so
    # formation() requires explicit positions for that pitch type
    if name == 'statsbomb':
        formation_kwargs = {'positions': [1, 2, 3, 5, 6, 12, 13, 15, 16, 22, 24]}
    else:
        formation_kwargs = {}
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 5))

    pitch = Pitch(pitch_type=pitch_type, **kwargs)
    pitch.draw(ax=axs[0, 0])
    pitch.formation('442', kind='scatter', s=100, color='red', ax=axs[0, 0],
                    **formation_kwargs)

    # flipped to the right-hand side
    pitch.draw(ax=axs[0, 1])
    pitch.formation('442', kind='scatter', flip=True, s=100, color='blue',
                    ax=axs[0, 1], **formation_kwargs)

    pitch_vertical = VerticalPitch(pitch_type=pitch_type, **kwargs)
    pitch_vertical.draw(ax=axs[1, 0])
    pitch_vertical.formation('442', kind='scatter', s=100, color='red',
                             ax=axs[1, 0], **formation_kwargs)

    # compressed into one half on a half pitch: flip=True is needed too,
    # because half=True alone uses the defensive half, which a half pitch
    # does not display
    pitch_half = VerticalPitch(pitch_type=pitch_type, half=True, **kwargs)
    pitch_half.draw(ax=axs[1, 1])
    pitch_half.formation('442', kind='scatter', half=True, flip=True, s=100,
                         color='blue', ax=axs[1, 1], **formation_kwargs)
    return fig


def test_formation_custom_dims_unsupported():
    """formation() looks up positions via getattr(position, pitch_type), so
    a custom dimensions object (rather than a named pitch type) raises."""
    pitch_type, kwargs = pitch_spec('center_scale')
    pitch = Pitch(pitch_type=pitch_type, **kwargs)
    fig, ax = pitch.draw(figsize=(4, 3))
    with pytest.raises(TypeError):
        pitch.formation('442', kind='scatter', ax=ax)


def test_get_formation_and_positions():
    pitch = Pitch()
    formation = pitch.get_formation('442')
    assert len(formation) == 11
    assert {position.name for position in formation} >= {'GK', 'RB', 'LB'}
    positions = pitch.get_positions()
    assert len(positions) > 0
    assert {'x', 'y'}.issubset(positions.columns)
    # every position must be inside the pitch extent
    extent = pitch.dim.pitch_extent
    assert positions['x'].between(extent[0], extent[1]).all()
    assert positions['y'].between(extent[2], extent[3]).all()
