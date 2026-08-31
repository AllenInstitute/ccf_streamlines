"""The shared voxel-matching helper.

This is the common core beneath the closest-streamline search, surface-voxel
collapsing, and 2-D coordinate calculation, and both open contributions reduce
to its behaviour. It is pure and takes small arrays, so it can be covered
exhaustively.

Its contract, stated plainly: given a lookup whose ``lookup_ind`` column is
sorted (or made sorted by a ``sorter``), return the ``ref_ind`` column of the
matching row for each query, and ``missing_value`` for queries with no match.
Ties in the key column resolve to whichever tied row the ordering placed first.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import _matching_voxel_indices


@pytest.fixture
def lookup():
    """Key column sorted and unique; value column deliberately unsorted."""
    return np.array([
        [10, 700],
        [20, 500],
        [30, 900],
        [40, 100],
    ])


def test_every_query_returns_its_matching_row(lookup):
    result = _matching_voxel_indices(np.array([10, 30, 40]), lookup)
    assert np.array_equal(result, np.array([700, 900, 100]))


def test_query_order_is_preserved(lookup):
    result = _matching_voxel_indices(np.array([40, 10, 30, 20]), lookup)
    assert np.array_equal(result, np.array([100, 700, 900, 500]))


def test_a_repeated_query_returns_the_same_answer_each_time(lookup):
    result = _matching_voxel_indices(np.array([20, 20, 20]), lookup)
    assert np.array_equal(result, np.array([500, 500, 500]))


def test_missing_keys_get_the_missing_value(lookup):
    """Zero by default, which is why flat index 0 must never be a real voxel."""
    result = _matching_voxel_indices(np.array([10, 15, 40]), lookup)
    assert np.array_equal(result, np.array([700, 0, 100]))


def test_the_missing_value_is_configurable(lookup):
    result = _matching_voxel_indices(np.array([15]), lookup, missing_value=-1)
    assert result[0] == -1


def test_all_missing(lookup):
    result = _matching_voxel_indices(np.array([1, 2, 3]), lookup, missing_value=-1)
    assert np.array_equal(result, np.array([-1, -1, -1]))


def test_an_empty_query_returns_an_empty_result(lookup):
    result = _matching_voxel_indices(np.array([], dtype=int), lookup)
    assert result.shape == (0,)


def test_columns_can_be_swapped(lookup):
    """The 2-D coordinate path searches column 1 and returns column 0."""
    result = _matching_voxel_indices(
        np.array([900, 100]), lookup, lookup_ind=1, ref_ind=0, missing_value=-1,
        sorter=np.argsort(lookup[:, 1]),
    )
    assert np.array_equal(result, np.array([30, 40]))


def test_a_sorter_makes_an_unsorted_key_column_searchable():
    """The view-lookup files are not sorted on their volume-index column."""
    unsorted = np.array([
        [1, 300],
        [2, 100],
        [3, 400],
        [4, 200],
    ])
    sorter = np.argsort(unsorted[:, 1])

    result = _matching_voxel_indices(
        np.array([100, 400]), unsorted, lookup_ind=1, ref_ind=0,
        missing_value=-1, sorter=sorter,
    )
    assert np.array_equal(result, np.array([2, 3]))


def test_without_a_sorter_an_unsorted_lookup_returns_wrong_answers_silently():
    """Characterization, not an endorsement.

    ``np.searchsorted`` assumes a sorted array and does not check. Nothing in
    the package validates the reference file's ordering, so an unsorted lookup
    yields wrong voxels rather than an error. Documented here so the assumption
    is visible.
    """
    unsorted = np.array([
        [1, 300],
        [2, 100],
        [3, 400],
        [4, 200],
    ])

    without_sorter = _matching_voxel_indices(
        np.array([100]), unsorted, lookup_ind=1, ref_ind=0, missing_value=-1
    )
    with_sorter = _matching_voxel_indices(
        np.array([100]), unsorted, lookup_ind=1, ref_ind=0, missing_value=-1,
        sorter=np.argsort(unsorted[:, 1]),
    )

    assert with_sorter[0] == 2
    assert without_sorter[0] != with_sorter[0]


def test_a_tie_resolves_to_the_first_row_in_sorter_order():
    """The behaviour issue #12 is about, in its smallest possible form.

    ``np.searchsorted`` returns the *left* insertion point, so a tied key picks
    whichever tied row the ordering placed first. The helper is not wrong; the
    caller choosing an unstable ordering is.
    """
    tied = np.array([
        [7, 100],
        [8, 100],
        [9, 200],
    ])

    stable = _matching_voxel_indices(
        np.array([100]), tied, lookup_ind=1, ref_ind=0, missing_value=-1,
        sorter=np.argsort(tied[:, 1], kind="stable"),
    )
    reversed_ties = _matching_voxel_indices(
        np.array([100]), tied, lookup_ind=1, ref_ind=0, missing_value=-1,
        sorter=np.lexsort((-np.arange(len(tied)), tied[:, 1])),
    )

    assert stable[0] == 7
    assert reversed_ties[0] == 8


def test_the_result_dtype_is_integer(lookup):
    result = _matching_voxel_indices(np.array([10]), lookup)
    assert np.issubdtype(result.dtype, np.integer)


def test_the_result_has_the_shape_of_the_query(lookup):
    query = np.array([10, 20, 30, 40, 50])
    assert _matching_voxel_indices(query, lookup).shape == query.shape
