import numpy as np
import pytest

import ccf_streamlines.coordinates as coordinates


def test_mismatch():
    test_coords = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    resolution = (10, 10, 10)

    with pytest.raises(ValueError):
        coordinates.coordinates_to_voxels(test_coords, resolution)


def test_coords_to_voxels():
    test_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [10.0, 10.0, 10.0],
            [15.0, 25.0, 0.0],
        ]
    )
    expected_voxels = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 2, 0],
        ]
    )

    resolution = (10, 10, 10)

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords, resolution) == expected_voxels
    )

    # Try doubling coordinates and resolution;
    # should get same answer
    double_resolution = (20, 20, 20)
    assert np.all(
        coordinates.coordinates_to_voxels(test_coords * 2, double_resolution)
        == expected_voxels
    )


def test_default_resolution_is_ten_microns():
    test_coords = np.array([[0.0, 0.0, 0.0], [15.0, 25.0, 35.0]])

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords)
        == coordinates.coordinates_to_voxels(test_coords, (10, 10, 10))
    )


def test_anisotropic_resolution_is_applied_per_axis():
    test_coords = np.array([[100.0, 100.0, 100.0]])
    expected_voxels = np.array([[10, 5, 1]])

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords, (10, 20, 100)) == expected_voxels
    )


def test_coordinates_are_floored_not_rounded():
    # 19.9 microns is still inside the second 10-micron voxel
    test_coords = np.array([[9.9, 10.0, 19.9]])
    expected_voxels = np.array([[0, 1, 1]])

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords, (10, 10, 10)) == expected_voxels
    )


def test_negative_coordinates_floor_away_from_zero():
    # Flooring is toward negative infinity, so -0.1 microns is voxel -1, not 0.
    test_coords = np.array([[-0.1, -10.0, -25.0]])
    expected_voxels = np.array([[-1, -1, -3]])

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords, (10, 10, 10)) == expected_voxels
    )


def test_result_is_an_integer_array():
    test_coords = np.array([[0.0, 0.0, 0.0], [15.0, 25.0, 35.0]])
    voxels = coordinates.coordinates_to_voxels(test_coords, (10, 10, 10))

    assert np.issubdtype(voxels.dtype, np.integer)
    assert voxels.shape == test_coords.shape


def test_non_numeric_dtype():
    test_coords = np.array(
        [
            ["0", "0", "0"],
            ["1", "1", "1"],
        ]
    )
    resolution = (10, 10, 10)

    with pytest.raises(ValueError, match="numeric dtype"):
        coordinates.coordinates_to_voxels(test_coords, resolution)


def test_two_dimensional_coordinates_work_with_a_two_tuple():
    # The function is not hardcoded to three dimensions; it only requires that
    # `resolution` match the second dimension of `coords`.
    test_coords = np.array([[0.0, 0.0], [15.0, 25.0]])
    expected_voxels = np.array([[0, 0], [1, 2]])

    assert np.all(
        coordinates.coordinates_to_voxels(test_coords, (10, 10)) == expected_voxels
    )
