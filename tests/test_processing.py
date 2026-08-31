"""Path cleaning, against paths with known repeats."""

import numpy as np

from ccf_streamlines.processing import remove_duplicate_voxels_from_paths


def test_consecutive_duplicates_are_removed_and_the_row_is_refilled():
    """Duplicates collapse and the row is left-packed, padded back to width."""
    paths = np.array(
        [
            [5, 5, 7, 7, 9, 0, 0, 0],
            [2, 3, 4, 0, 0, 0, 0, 0],
        ]
    )

    result = remove_duplicate_voxels_from_paths(paths)

    assert result.shape == paths.shape
    assert np.array_equal(result[0], np.array([5, 7, 9, 0, 0, 0, 0, 0]))
    assert np.array_equal(result[1], np.array([2, 3, 4, 0, 0, 0, 0, 0]))


def test_a_path_with_no_duplicates_is_unchanged():
    paths = np.array([[11, 12, 13, 14, 0, 0]])
    assert np.array_equal(remove_duplicate_voxels_from_paths(paths), paths)


def test_only_consecutive_duplicates_are_removed():
    """A voxel repeated non-consecutively is a genuine revisit, and is kept."""
    paths = np.array([[4, 4, 8, 4, 0, 0]])
    result = remove_duplicate_voxels_from_paths(paths)
    assert np.array_equal(result[0], np.array([4, 8, 4, 0, 0, 0]))


def test_a_long_run_collapses_to_one_voxel():
    paths = np.array([[6, 6, 6, 6, 6, 0]])
    result = remove_duplicate_voxels_from_paths(paths)
    assert np.array_equal(result[0], np.array([6, 0, 0, 0, 0, 0]))


def test_rows_are_cleaned_independently():
    paths = np.array(
        [
            [1, 1, 1, 2, 0, 0],
            [3, 4, 5, 6, 7, 0],
        ]
    )
    result = remove_duplicate_voxels_from_paths(paths)
    assert np.array_equal(result[0], np.array([1, 2, 0, 0, 0, 0]))
    assert np.array_equal(result[1], np.array([3, 4, 5, 6, 7, 0]))


def test_input_must_be_zero_padded():
    """A row that fills its full width loses its last voxel.

    The function detects voxels by looking at where the row *changes*, so the
    final voxel is only seen because the padding after it differs. The real
    ``paths`` dataset is always zero-padded, so this never bites in practice --
    but it is a precondition, not a free choice, and a caller building paths by
    hand needs to know.
    """
    unpadded = np.array([[5, 5, 7, 7, 9, 9]])
    result = remove_duplicate_voxels_from_paths(unpadded)
    assert 9 not in result[0].tolist()

    padded = np.array([[5, 5, 7, 7, 9, 9, 0]])
    assert 9 in remove_duplicate_voxels_from_paths(padded)[0].tolist()


def test_matches_the_inline_deduplication_in_metrics():
    """`metrics.measure_streamline_layer_thicknesses` re-implements this with a
    Python loop instead of calling it. The two must agree, or layer depths are
    measured against different paths than everything else."""
    paths = np.array(
        [
            [5, 5, 7, 7, 9, 0, 0, 0],
            [2, 3, 3, 0, 0, 0, 0, 0],
        ]
    )

    vectorized = remove_duplicate_voxels_from_paths(paths.copy())

    # The loop from metrics.py, verbatim.
    inline = np.zeros_like(paths)
    paths_diff = np.diff(paths, axis=1)
    for i in range(paths.shape[0]):
        unique_inds = np.flatnonzero(paths_diff[i, :])
        inline[i, : len(unique_inds)] = paths[i, :][unique_inds]

    assert np.array_equal(vectorized, inline)
