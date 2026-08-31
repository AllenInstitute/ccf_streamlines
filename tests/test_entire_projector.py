"""``IsocortexEntireProjector``: one value per streamline, and their ordering."""

import numpy as np
import pytest

from ccf_streamlines.projection import IsocortexEntireProjector


@pytest.fixture
def projector(mini_ccf):
    return IsocortexEntireProjector(
        mini_ccf.surface_paths_file, resolution=mini_ccf.resolution
    )


@pytest.fixture
def ramp_volume(mini_ccf):
    volume = mini_ccf.volume()
    for step, voxel in enumerate(mini_ccf.path_voxels(0)):
        volume[tuple(voxel)] = float(step + 1)
    return volume


def test_one_value_per_streamline_in_file_order(projector, mini_ccf, ramp_volume):
    """Unlike the view projectors, paths are not reordered: every streamline in
    the file gets a value, in the file's own order."""
    values = projector.project_volume(ramp_volume, kind="max")

    assert values.shape == (mini_ccf.paths.shape[0],)
    assert values[0] == 8.0
    assert np.all(values[1:] == 0.0)


@pytest.mark.parametrize(
    "kind,expected", [("max", 8.0), ("min", 1.0), ("mean", 4.5), ("average", 4.5), ("sum", 36.0)]
)
def test_each_aggregation(projector, mini_ccf, ramp_volume, kind, expected):
    values = projector.project_volume(ramp_volume, kind=kind)
    assert values[0] == pytest.approx(expected)


def test_streamlines_absent_from_any_view_are_still_projected(
    projector, mini_ccf
):
    """This class covers every streamline, including those no view uses."""
    path_index = int(mini_ccf.out_of_view_path_indices[0])
    volume = mini_ccf.volume()
    volume[tuple(mini_ccf.path_voxels(path_index)[2])] = 4.0

    values = projector.project_volume(volume, kind="max")

    assert values[path_index] == 4.0


# -- top_of_streamline_coords ----------------------------------------------


def test_top_coordinates_are_the_first_voxel_of_each_streamline(projector, mini_ccf):
    coords = projector.top_of_streamline_coords()

    assert coords.shape == (mini_ccf.paths.shape[0], 3)
    for i in range(mini_ccf.paths.shape[0]):
        assert np.array_equal(coords[i], mini_ccf.path_voxels(i)[0])


def test_top_coordinates_are_ordered_to_match_project_volume(
    projector, mini_ccf
):
    """The documented correspondence: element *i* of the projected values is
    the streamline whose top is row *i* here.

    Verified by lighting one streamline at a time and checking that the single
    non-zero value lands at the row whose top coordinate is that streamline's.
    """
    coords = projector.top_of_streamline_coords()

    for path_index in (0, 5, int(mini_ccf.out_of_view_path_indices[0])):
        volume = mini_ccf.volume()
        volume[tuple(mini_ccf.path_voxels(path_index)[2])] = 9.0

        values = projector.project_volume(volume, kind="max")
        lit = np.flatnonzero(values)

        assert lit.tolist() == [path_index]
        assert np.array_equal(coords[lit[0]], mini_ccf.path_voxels(path_index)[0])


def test_top_coordinates_in_microns(projector, mini_ccf):
    in_voxels = projector.top_of_streamline_coords(scale="voxels")
    in_microns = projector.top_of_streamline_coords(scale="microns")

    assert np.array_equal(in_microns, in_voxels * np.array(mini_ccf.resolution))


def test_the_default_scale_is_voxels(projector):
    assert np.array_equal(
        projector.top_of_streamline_coords(),
        projector.top_of_streamline_coords(scale="voxels"),
    )


# -- documented error conditions -------------------------------------------


def test_a_volume_of_the_wrong_shape_raises(projector):
    with pytest.raises(ValueError, match="must match lookup volume shape"):
        projector.project_volume(np.zeros((2, 2, 2)))


def test_a_volume_of_a_non_numeric_dtype_raises(projector, mini_ccf):
    with pytest.raises(ValueError, match="integer or float"):
        projector.project_volume(mini_ccf.volume(dtype=bool))


# -- defects ---------------------------------------------------------------


def test_an_unrecognised_kind_raises_a_clear_error(projector, ramp_volume):
    with pytest.raises(ValueError):
        projector.project_volume(ramp_volume, kind="maximum")


def test_an_unrecognised_scale_raises(projector):
    """Currently returns None, so the caller gets a TypeError somewhere else."""
    with pytest.raises(ValueError):
        projector.top_of_streamline_coords(scale="mm")


@pytest.mark.parametrize("kind", ["max", "min"])
def test_projection_does_not_modify_the_callers_volume(projector, mini_ccf, kind):
    volume = mini_ccf.volume()
    volume[tuple(mini_ccf.path_voxels(0)[2])] = 3.0
    before = volume.copy()

    projector.project_volume(volume, kind=kind)

    assert np.array_equal(volume, before)
