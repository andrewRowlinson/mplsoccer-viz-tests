"""annotate and text: no image comparison here - the coordinate handling
is what mplsoccer adds, so that is what is tested. mplsoccer must swap
x and y into display coordinates on a vertical pitch, for the text
position and for both ends of an annotation arrow."""
import matplotlib.text
from mplsoccer import Pitch, VerticalPitch


def test_annotate():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    annotation = pitch.annotate('a note', xy=(60, 20), xytext=(30, 10),
                                arrowprops={'arrowstyle': '->'}, ax=ax)
    assert isinstance(annotation, matplotlib.text.Annotation)
    assert annotation.xy == (60, 20)
    assert annotation.get_position() == (30, 10)


def test_annotate_vertical_swaps_coordinates():
    pitch_vertical = VerticalPitch()
    fig, ax = pitch_vertical.draw(figsize=(3, 4))
    annotation = pitch_vertical.annotate('a note', xy=(60, 20), xytext=(30, 10),
                                         arrowprops={'arrowstyle': '->'}, ax=ax)
    assert isinstance(annotation, matplotlib.text.Annotation)
    assert annotation.xy == (20, 60)
    assert annotation.get_position() == (10, 30)


def test_text():
    pitch = Pitch()
    fig, ax = pitch.draw(figsize=(4, 3))
    text = pitch.text(60, 20, 'a note', ax=ax)
    assert isinstance(text, matplotlib.text.Text)
    assert text.get_position() == (60, 20)


def test_text_vertical_swaps_coordinates():
    pitch_vertical = VerticalPitch()
    fig, ax = pitch_vertical.draw(figsize=(3, 4))
    text = pitch_vertical.text(60, 20, 'a note', ax=ax)
    assert isinstance(text, matplotlib.text.Text)
    assert text.get_position() == (20, 60)
