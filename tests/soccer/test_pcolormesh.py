"""pcolormesh overlays of a dense surface on the pitch (mplsoccer 1.8.1).

The surface is a Gaussian bump centred on a fixed physical spot
(three quarters along the pitch, three tenths up as displayed) and a
red marker is scattered at the same spot. In every image the marker
should sit on the bright peak of the bump: a bump in the wrong place
or the wrong orientation indicates a bug. The bump is computed from
the cell centres in pitch coordinates, so on inverted-y pitch types
(statsbomb, wyscout, metricasports) the surface rows follow the
inverted axis and the picture stays physically identical."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, mirror_inverted_y, PITCH_TYPES, pitch_spec, scale_pad

MPL = mpl_kwargs('pcolormesh')

# physical position of the bump as fractions of the pitch (x along the
# pitch, y up the pitch as displayed for a horizontal pitch)
BUMP_X, BUMP_Y = 0.75, 0.3


def bump_point(dim):
    """The bump centre in pitch coordinates."""
    xmin, xmax, ymin, ymax = dim.pitch_extent
    x = xmin + BUMP_X * (xmax - xmin)
    y = mirror_inverted_y(dim, ymin + BUMP_Y * (ymax - ymin))
    return x, y


def bump_surface(dim, extent, shape):
    """A Gaussian bump on a (ny, nx) grid spanning the extent, centred on
    the bump point, with a width of 15% of the pitch in each direction."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = shape
    x_centres = xmin + (np.arange(nx) + 0.5) / nx * (xmax - xmin)
    y_centres = ymin + (np.arange(ny) + 0.5) / ny * (ymax - ymin)
    x_grid, y_grid = np.meshgrid(x_centres, y_centres)
    px, py = bump_point(dim)
    sx = 0.15 * (dim.pitch_extent[1] - dim.pitch_extent[0])
    sy = 0.15 * (dim.pitch_extent[3] - dim.pitch_extent[2])
    return np.exp(-(((x_grid - px) / sx) ** 2 + ((y_grid - py) / sy) ** 2))


def draw_pair(pitch, pitch_vertical, extent=None, shape=(32, 48), **kwargs):
    """Draw the surface and marker on a horizontal and vertical pitch side by side."""
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    px, py = bump_point(pitch.dim)
    surface_extent = pitch.dim.pitch_extent if extent is None else extent
    surface = bump_surface(pitch.dim, surface_extent, shape)
    for p, axis in zip((pitch, pitch_vertical), ax):
        p.draw(axis)
        p.pcolormesh(surface, extent=extent, ax=axis, cmap='viridis', **kwargs)
        p.scatter(px, py, ax=axis, color='red', zorder=3)
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pcolormesh(name):
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    return draw_pair(pitch, pitch_vertical)


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pcolormesh_half(name):
    """A full-pitch surface on a half pitch: the hidden half is clipped."""
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, half=True, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, half=True,
                                   **kwargs)
    return draw_pair(pitch, pitch_vertical)


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pcolormesh_extent(name):
    """A coarse surface over the final third and the middle half of the
    pitch width only, with cell edges drawn to show the grid."""
    pitch_type, kwargs = pitch_spec(name)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    xmin, xmax, ymin, ymax = pitch.dim.pitch_extent
    extent = (xmin + 2 / 3 * (xmax - xmin), xmax,
              ymin + 0.25 * (ymax - ymin), ymin + 0.75 * (ymax - ymin))
    return draw_pair(pitch, pitch_vertical, extent=extent, shape=(4, 6),
                     edgecolors='white', linewidth=0.5)


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_pcolormesh_pad(name):
    """Negative padding on all sides (the same as test_pad_clip's negative
    pad tests): the surface is clipped to the shrunken axes like a heatmap,
    and the marker stays on the peak. The amount clipped differs between
    pitch types because the pad is in pitch units, so compare each image
    with the matching test_negative_pad_heatmap baseline."""
    pitch_type, kwargs = pitch_spec(name)
    pad = dict.fromkeys(['pad_left', 'pad_right', 'pad_bottom', 'pad_top'],
                        scale_pad(-15, name))
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **pad, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **pad, **kwargs)
    return draw_pair(pitch, pitch_vertical)
