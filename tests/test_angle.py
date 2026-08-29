"""Affine construction, nearest-streamline lookup, and streamline-plane angle.

The angle cases are anchored at values that can be verified by inspection:
a streamline perpendicular to the plane is 90 degrees, one lying in the plane
is 0, and one at 45 degrees is 45.
"""

import numpy as np
import pytest

from ccf_streamlines.angle import (
    determine_angle_between_streamline_and_plane,
    find_closest_streamline,
    vector_to_3d_affine_matrix,
)

RESOLUTION_NOT_FORWARDED = (
    "`find_closest_streamline` accepts `resolution` but does not forward it to "
    "`coordinates_to_voxels`, so a non-default resolution voxelises against "
    "(10, 10, 10); remove this marker when it is fixed"
)

#: Maps the unit square onto the xy-plane, so the plane normal is +z.
XY_PLANE = vector_to_3d_affine_matrix([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])


# -- vector_to_3d_affine_matrix --------------------------------------------


def test_affine_matrix_layout():
    """The first nine entries are the 3x3 basis, the last three the translation."""
    M = vector_to_3d_affine_matrix(list(range(12)))

    assert M.shape == (3, 4)
    assert np.array_equal(M[:, :3], np.arange(9).reshape(3, 3))
    assert np.array_equal(M[:, 3], np.array([9, 10, 11]))


def test_affine_matrix_translates_the_origin():
    M = vector_to_3d_affine_matrix([1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 6, 7])
    assert np.array_equal(M @ np.array([0, 0, 0, 1]), np.array([5, 6, 7]))


# -- determine_angle_between_streamline_and_plane --------------------------


def test_streamline_perpendicular_to_the_plane_is_ninety_degrees():
    """Running along +z, the xy-plane's normal."""
    streamline = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    angle = determine_angle_between_streamline_and_plane(streamline, XY_PLANE)
    assert angle == pytest.approx(90.0)


@pytest.mark.parametrize(
    "direction", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
)
def test_streamline_lying_in_the_plane_is_zero_degrees(direction):
    streamline = np.array([direction, [0.0, 0.0, 0.0]])
    angle = determine_angle_between_streamline_and_plane(streamline, XY_PLANE)
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_streamline_at_forty_five_degrees():
    streamline = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    angle = determine_angle_between_streamline_and_plane(streamline, XY_PLANE)
    assert angle == pytest.approx(45.0)


def test_only_the_endpoints_of_the_streamline_matter():
    """The function uses the pia and white-matter ends, not the path between."""
    straight = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    wiggly = np.array([[0.0, 0.0, 10.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    assert determine_angle_between_streamline_and_plane(
        straight, XY_PLANE
    ) == pytest.approx(determine_angle_between_streamline_and_plane(wiggly, XY_PLANE))


def test_a_rotated_plane_rotates_the_angle():
    """Swap the plane to the xz-plane; a streamline along +z now lies in it."""
    xz_plane = vector_to_3d_affine_matrix([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    streamline = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    assert determine_angle_between_streamline_and_plane(
        streamline, xz_plane
    ) == pytest.approx(0.0, abs=1e-9)


def test_plane_translation_does_not_change_the_angle():
    translated = vector_to_3d_affine_matrix([1, 0, 0, 0, 1, 0, 0, 0, 1, 100, 200, 300])
    streamline = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    assert determine_angle_between_streamline_and_plane(
        streamline, translated
    ) == pytest.approx(90.0)


# -- find_closest_streamline -----------------------------------------------


def test_a_coordinate_on_a_streamline_returns_that_streamline(mini_ccf):
    path_index = 0
    coord = mini_ccf.coord_on_path(path_index, 3)

    result = find_closest_streamline(
        coord,
        mini_ccf.closest_surface_voxel_file,
        mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution,
        volume_shape=mini_ccf.volume_shape,
    )

    assert np.array_equal(result, mini_ccf.path_microns(path_index))


def test_the_reference_may_be_a_preloaded_array(mini_ccf):
    """The documented alternative to passing a file path."""
    import h5py

    with h5py.File(mini_ccf.closest_surface_voxel_file, "r") as f:
        closest = f["closest surface voxel"][:]

    coord = mini_ccf.coord_on_path(0, 3)
    from_array = find_closest_streamline(
        coord, closest, mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution, volume_shape=mini_ccf.volume_shape,
    )
    from_path = find_closest_streamline(
        coord, mini_ccf.closest_surface_voxel_file, mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution, volume_shape=mini_ccf.volume_shape,
    )
    assert np.array_equal(from_array, from_path)


def test_surface_paths_may_be_an_open_h5py_file(mini_ccf):
    import h5py

    coord = mini_ccf.coord_on_path(0, 3)
    with h5py.File(mini_ccf.surface_paths_file, "r") as f:
        from_handle = find_closest_streamline(
            coord, mini_ccf.closest_surface_voxel_file, f,
            resolution=mini_ccf.resolution, volume_shape=mini_ccf.volume_shape,
        )
    assert np.array_equal(from_handle, mini_ccf.path_microns(0))


def test_a_coordinate_outside_cortex_returns_an_empty_array(mini_ccf, caplog):
    """The dorso-ventral planes at each end have no streamline voxels."""
    outside = np.array([3.0, 0.0, 1.0]) * np.array(mini_ccf.resolution)

    result = find_closest_streamline(
        outside,
        mini_ccf.closest_surface_voxel_file,
        mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution,
        volume_shape=mini_ccf.volume_shape,
    )

    assert result.size == 0
    assert "not within isocortex" in caplog.text


def test_a_right_hemisphere_coordinate_comes_back_on_the_right(mini_ccf):
    """Reference data exists only on the left, so the lookup reflects, then
    reflects the answer back."""
    z_size = mini_ccf.volume_shape[2]
    left_voxel = mini_ccf.path_voxels(0)[3]
    right_voxel = left_voxel.copy()
    right_voxel[2] = z_size - left_voxel[2]
    coord = right_voxel * np.array(mini_ccf.resolution)

    result = find_closest_streamline(
        coord,
        mini_ccf.closest_surface_voxel_file,
        mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution,
        volume_shape=mini_ccf.volume_shape,
    )

    expected = mini_ccf.path_voxels(0).copy()
    expected[:, 2] = z_size - expected[:, 2]
    assert np.array_equal(result, expected * np.array(mini_ccf.resolution))


def test_a_coordinate_may_be_given_as_a_flat_triple(mini_ccf):
    coord = mini_ccf.coord_on_path(0, 3)
    flat = find_closest_streamline(
        coord, mini_ccf.closest_surface_voxel_file, mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution, volume_shape=mini_ccf.volume_shape,
    )
    nested = find_closest_streamline(
        coord.reshape(1, 3), mini_ccf.closest_surface_voxel_file,
        mini_ccf.surface_paths_file,
        resolution=mini_ccf.resolution, volume_shape=mini_ccf.volume_shape,
    )
    assert np.array_equal(flat, nested)


@pytest.mark.xfail(strict=True, reason=RESOLUTION_NOT_FORWARDED)
def test_resolution_is_honoured_when_finding_the_streamline(mini_ccf):
    """The same physical point, expressed at a coarser voxel size.

    ``resolution`` is used to scale the *returned* coordinates but not to
    convert the *input* coordinate to a voxel, so at any resolution other than
    (10, 10, 10) the wrong voxel is looked up. Here it lands outside the
    lookup entirely and an empty array comes back.
    """
    resolution = (20, 20, 20)
    voxel = mini_ccf.path_voxels(0)[3]
    coord = voxel * np.array(resolution)

    result = find_closest_streamline(
        coord,
        mini_ccf.closest_surface_voxel_file,
        mini_ccf.surface_paths_file,
        resolution=resolution,
        volume_shape=mini_ccf.volume_shape,
    )

    assert result.size > 0
    assert np.array_equal(result, mini_ccf.path_voxels(0) * np.array(resolution))
