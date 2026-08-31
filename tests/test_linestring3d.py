"""Streamline geometry, against analytically known cases.

Every depth calculation in the package is built on this class, so the cases
here are chosen so the right answer can be worked out by hand: a 3-4-5
triangle, an axis-aligned path, a point whose perpendicular offset is exact.
"""

import numpy as np
import pytest

from ccf_streamlines.linestring3d import LineString3D


@pytest.fixture
def straight_path():
    """Ten units long, along +y, so distances along it are just y."""
    return LineString3D(np.array([[0.0, y, 0.0] for y in range(11)]))


@pytest.fixture
def bent_path():
    """Two segments: 3 along x, then 4 along y. Total length 7."""
    return LineString3D(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))


def test_segment_lengths(bent_path):
    assert np.array_equal(bent_path.segment_lengths(), np.array([3.0, 4.0]))


def test_length_is_the_sum_of_segments(bent_path, straight_path):
    assert bent_path.length == 7.0
    assert straight_path.length == 10.0


def test_segment_lengths_of_a_diagonal():
    """A 3-4-5 triangle, so the hypotenuse is exactly 5."""
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    assert path.segment_lengths() == pytest.approx([5.0])
    assert path.length == pytest.approx(5.0)


# -- project ---------------------------------------------------------------


@pytest.mark.parametrize("y", [0.0, 1.0, 5.5, 10.0])
def test_project_a_point_lying_on_the_path(straight_path, y):
    assert straight_path.project(np.array([0.0, y, 0.0])) == pytest.approx(y)


def test_project_a_point_beside_the_path(straight_path):
    """Distance along is unaffected by perpendicular displacement."""
    assert straight_path.project(np.array([3.0, 4.0, 2.0])) == pytest.approx(4.0)


def test_project_normalized_is_the_fraction_of_total_length(straight_path):
    assert straight_path.project(
        np.array([0.0, 2.5, 0.0]), normalized=True
    ) == pytest.approx(0.25)
    assert straight_path.project(
        np.array([0.0, 10.0, 0.0]), normalized=True
    ) == pytest.approx(1.0)


def test_project_across_a_bend(bent_path):
    """Three units along the first segment, then two into the second."""
    assert bent_path.project(np.array([3.0, 2.0, 0.0])) == pytest.approx(5.0)


def test_project_before_the_start_clamps_to_zero(straight_path):
    """A point behind the pia end does not project onto any segment."""
    assert straight_path.project(np.array([0.0, -5.0, 0.0])) == pytest.approx(0.0)


def test_project_past_the_end_clamps_to_the_last_vertex(straight_path):
    """Past the white-matter end, the nearest vertex is the last one.

    Note the returned value is the length up to that vertex, which for the
    final vertex is the whole path.
    """
    assert straight_path.project(np.array([0.0, 50.0, 0.0])) == pytest.approx(10.0)


# -- offset_of_point -------------------------------------------------------


def test_offset_of_a_point_on_the_path_is_zero(straight_path):
    """The property that makes exact integer assertions possible elsewhere."""
    offset = straight_path.offset_of_point(np.array([0.0, 4.0, 0.0]))
    assert np.array_equal(offset, np.zeros(3))


def test_offset_is_the_perpendicular_displacement(straight_path):
    offset = straight_path.offset_of_point(np.array([2.0, 4.0, -3.0]))
    assert offset == pytest.approx([2.0, 0.0, -3.0])


def test_offset_of_a_point_beyond_the_end_is_measured_from_the_last_vertex(
    straight_path,
):
    offset = straight_path.offset_of_point(np.array([1.0, 20.0, 0.0]))
    assert offset == pytest.approx([1.0, 10.0, 0.0])


def test_offset_across_a_bend(bent_path):
    offset = bent_path.offset_of_point(np.array([3.0, 2.0, 5.0]))
    assert offset == pytest.approx([0.0, 0.0, 5.0])


# -- rotation_to_vector ----------------------------------------------------


def test_rotation_aligns_the_path_with_the_target_vector():
    """A path along +x, rotated to point along +y."""
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    rot = path.rotation_to_vector(np.array([0.0, 1.0, 0.0]))

    rotated_end = rot @ path.coords[-1, :]
    assert rotated_end == pytest.approx([0.0, 5.0, 0.0])


def test_rotation_is_orthonormal_and_preserves_length():
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
    rot = path.rotation_to_vector(np.array([0.0, 1.0, 0.0]))

    assert rot @ rot.T == pytest.approx(np.identity(3))
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert np.linalg.norm(rot @ path.coords[-1, :]) == pytest.approx(np.sqrt(14))


def test_rotation_to_an_already_aligned_vector_is_the_identity():
    """The mini-CCF's streamlines run straight down +y, so this is the case
    the coordinate projector actually hits, and it must be exact."""
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]]))
    rot = path.rotation_to_vector(np.array([0.0, 1.0, 0.0]))
    assert np.array_equal(rot, np.identity(3))


def test_rotation_target_need_not_be_unit_length():
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert np.array_equal(
        path.rotation_to_vector(np.array([0.0, 1.0, 0.0])),
        path.rotation_to_vector(np.array([0.0, 9.0, 0.0])),
    )


def test_rotation_to_an_antiparallel_vector_is_undefined():
    """Characterization, not an assertion that this is correct.

    The Rodrigues construction divides by ``1 + dot``, which is zero when the
    path points exactly opposite the target. Cortical streamlines always run
    roughly pia-to-white-matter, so the coordinate projector never asks for
    this -- but a caller using `LineString3D` directly can, and gets
    non-finite values rather than an error.
    """
    path = LineString3D(np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rot = path.rotation_to_vector(np.array([0.0, 1.0, 0.0]))
    assert not np.all(np.isfinite(rot))
