"""Polygons, voronoi diagrams and goal angles."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mplsoccer import Pitch, VerticalPitch

from helpers import mpl_kwargs, PITCH_TYPES, pitch_spec, random_points

MPL = mpl_kwargs('polygons_voronoi')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_polygon(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 5, rng)
    verts = [np.stack([x, y], axis=1)]

    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])
    pitch.polygon(verts, fc='blue', alpha=0.7, ec='black', ax=ax[0])
    pitch.scatter(x, y, color='red', ax=ax[0])
    pitch_vertical.polygon(verts, fc='blue', alpha=0.7, ec='black', ax=ax[1])
    pitch_vertical.scatter(x, y, color='red', ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_voronoi(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 22, rng)
    teams = np.array([0] * 11 + [1] * 11)

    pitch.draw(ax=ax[0])
    pitch_vertical.draw(ax=ax[1])

    team1, team2 = pitch.voronoi(x, y, teams)
    pitch.polygon(team1, color='blue', alpha=0.4, ax=ax[0])
    pitch.polygon(team2, color='red', alpha=0.4, ax=ax[0])
    pitch.scatter(x[teams == 1], y[teams == 1], color='blue', ax=ax[0])
    pitch.scatter(x[teams == 0], y[teams == 0], color='red', ax=ax[0])

    team1_vertical, team2_vertical = pitch_vertical.voronoi(x, y, teams)
    pitch_vertical.polygon(team1_vertical, color='blue', alpha=0.4, ax=ax[1])
    pitch_vertical.polygon(team2_vertical, color='red', alpha=0.4, ax=ax[1])
    pitch_vertical.scatter(x[teams == 1], y[teams == 1], color='blue', ax=ax[1])
    pitch_vertical.scatter(x[teams == 0], y[teams == 0], color='red', ax=ax[1])
    return fig


def goal_angle_figure(name, goal):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 2, rng)

    pitch.draw(ax[0])
    pitch_vertical.draw(ax[1])
    pitch.goal_angle(x, y, ax=ax[0], goal=goal)
    pitch_vertical.goal_angle(x, y, ax=ax[1], goal=goal)
    pitch.scatter(x, y, color='red', ax=ax[0])
    pitch_vertical.scatter(x, y, color='red', ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_goal_angle_right(name):
    return goal_angle_figure(name, goal='right')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_goal_angle_left(name):
    return goal_angle_figure(name, goal='left')


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_convexhull(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 15, rng)

    pitch.draw(ax[0])
    hull = pitch.convexhull(x, y)
    pitch.polygon(hull, ec='blue', fc='blue', alpha=0.3, ax=ax[0])
    pitch.scatter(x, y, color='red', ax=ax[0])

    pitch_vertical.draw(ax[1])
    hull_vertical = pitch_vertical.convexhull(x, y)
    pitch_vertical.polygon(hull_vertical, ec='blue', fc='blue', alpha=0.3, ax=ax[1])
    pitch_vertical.scatter(x, y, color='red', ax=ax[1])
    return fig


@pytest.mark.parametrize('name', PITCH_TYPES)
@pytest.mark.mpl_image_compare(**MPL)
def test_triplot(name):
    pitch_type, kwargs = pitch_spec(name)
    rng = np.random.default_rng(42)
    pitch = Pitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    pitch_vertical = VerticalPitch(pitch_type=pitch_type, line_zorder=2, **kwargs)
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
    x, y = random_points(pitch.dim, 15, rng)

    pitch.draw(ax[0])
    pitch.triplot(x, y, color='blue', ax=ax[0])
    pitch.scatter(x, y, color='red', zorder=3, ax=ax[0])

    pitch_vertical.draw(ax[1])
    pitch_vertical.triplot(x, y, color='blue', ax=ax[1])
    pitch_vertical.scatter(x, y, color='red', zorder=3, ax=ax[1])
    return fig
