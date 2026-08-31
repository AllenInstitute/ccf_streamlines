"""The ISH upsampling helper.

Always called with a small ``target_volume_shape``: the default allocates a
1320 x 800 x 1140 float array, several gigabytes, which no test should do.
"""

import numpy as np
import pytest

from ccf_streamlines.dataset import upscale_ish_volume


@pytest.fixture
def small_volume():
    """Shape (3, 1, 2) with distinct values, so any axis mixup is visible."""
    return np.arange(6, dtype=float).reshape(3, 1, 2)


def test_each_target_voxel_takes_its_downscaled_source_value(small_volume):
    """With a 2x ratio, target voxel (i, j, k) comes from source (i//2, j//2, k//2)."""
    # rotate_axes swaps 0 and 2, so a (3, 1, 2) input becomes (2, 1, 3).
    result = upscale_ish_volume(
        small_volume,
        orig_voxel_size=20,
        target_voxel_size=10,
        target_volume_shape=(4, 2, 6),
    )
    swapped = np.swapaxes(small_volume, 0, 2)

    assert result.shape == (4, 2, 6)
    for i in range(4):
        for j in range(2):
            for k in range(6):
                assert result[i, j, k] == swapped[i // 2, j // 2, k // 2]


def test_rotate_axes_swaps_the_first_and_last_axes(small_volume):
    """Volumes from the ISH atlas API have anterior-posterior in z and
    left-right in x; the CCF has those swapped."""
    rotated = upscale_ish_volume(
        small_volume,
        orig_voxel_size=10,
        target_voxel_size=10,
        target_volume_shape=(2, 1, 3),
        rotate_axes=True,
    )
    unrotated = upscale_ish_volume(
        np.swapaxes(small_volume, 0, 2),
        orig_voxel_size=10,
        target_voxel_size=10,
        target_volume_shape=(2, 1, 3),
        rotate_axes=False,
    )
    assert np.array_equal(rotated, unrotated)


def test_without_rotation_a_matching_shape_round_trips(small_volume):
    """A 1:1 ratio and a matching target shape is the identity."""
    result = upscale_ish_volume(
        small_volume,
        orig_voxel_size=10,
        target_voxel_size=10,
        target_volume_shape=small_volume.shape,
        rotate_axes=False,
    )
    assert np.array_equal(result, small_volume)


def test_upscaling_repeats_each_source_voxel_ratio_times():
    """A single source voxel fills a ratio-cubed block of the target."""
    volume = np.array([[[7.0]]])
    result = upscale_ish_volume(
        volume,
        orig_voxel_size=30,
        target_voxel_size=10,
        target_volume_shape=(3, 3, 3),
        rotate_axes=False,
    )
    assert np.array_equal(result, np.full((3, 3, 3), 7.0))


def test_target_larger_than_the_scaled_source_raises(small_volume):
    """Asking for more target voxels than the source can cover is an
    out-of-bounds index, not a silent zero-fill."""
    with pytest.raises(IndexError):
        upscale_ish_volume(
            small_volume,
            orig_voxel_size=20,
            target_voxel_size=10,
            target_volume_shape=(100, 2, 6),
        )
