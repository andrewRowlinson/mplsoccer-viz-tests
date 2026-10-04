"""Shared helpers for the soccer visual tests.

Each test is a self-contained, hand-inspected example. The image tests are compared against baseline
images with pytest-mpl. Text is stripped (``remove_text=True``) so the
comparison does not depend on font rendering, and a low dpi keeps the
baseline images lightweight.
"""
import numpy as np
from mplsoccer.soccer.dimensions import center_scale_dims, size_varies, valid

def mpl_kwargs(group):
    """Keyword arguments for @pytest.mark.mpl_image_compare.

    Each test module passes its own group name so its baseline images live
    in a separate directory (tests/soccer/baseline/<group>/). Regenerate
    them with generate_baselines.py, which runs pytest per test module."""
    return {'remove_text': True,
            'savefig_kwargs': {'dpi': 60},
            'baseline_dir': f'baseline/{group}'}

# all pitch types under test: the valid named pitch types plus a custom
# center-scaled dimension object
PITCH_TYPES = valid + ['center_scale']


def pitch_spec(name):
    """Return (pitch_type, pitch_kwargs) for a pitch-type name."""
    if name == 'center_scale':
        dim = center_scale_dims(pitch_width=68, pitch_length=105,
                                width=5, length=100, invert_y=False)
        return dim, {}
    if name in size_varies:
        return name, {'pitch_length': 105, 'pitch_width': 68}
    return name, {}


def scale_pad(pad, name):
    """Scale a pad value (in metres-like units) to the pitch coordinate system."""
    if name == 'metricasports':
        return pad / 100
    if name == 'center_scale':
        return pad / 10
    return pad


def mirror_inverted_y(dim, y):
    """Mirror y values on invert_y pitch types (statsbomb, wyscout,
    metricasports) so the same random draws land at the same physical
    spots as on the non-inverted types. This makes every pitch type's
    baseline image show the same pattern, which is much easier to
    hand-validate: any image that doesn't look like a scaled copy of
    its neighbours indicates a bug."""
    if dim.invert_y:
        return dim.pitch_extent[2] + dim.pitch_extent[3] - y
    return y


def random_points(dim, size, rng):
    """Random x, y points uniformly drawn from the pitch extent.

    The y values are mirrored on invert_y pitch types so all pitch
    types show the same physical pattern."""
    x = rng.uniform(low=dim.pitch_extent[0], high=dim.pitch_extent[1], size=size)
    y = rng.uniform(low=dim.pitch_extent[2], high=dim.pitch_extent[3], size=size)
    return x, mirror_inverted_y(dim, y)
