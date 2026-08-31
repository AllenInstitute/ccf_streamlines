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


def test_without_a_sorter_an_unsorted_lookup_raises():
    """This used to return a wrong voxel, silently.

    ``np.searchsorted`` assumes a sorted array and does not verify it, so an
    out-of-order reference file produced wrong matches rather than an error.
    The same query is fine once a ``sorter`` supplies the ordering.
    """
    unsorted = np.array([
        [1, 300],
        [2, 100],
        [3, 400],
        [4, 200],
    ])

    with pytest.raises(ValueError, match="must increase monotonically"):
        _matching_voxel_indices(
            np.array([100]), unsorted, lookup_ind=1, ref_ind=0, missing_value=-1
        )

    with_sorter = _matching_voxel_indices(
        np.array([100]), unsorted, lookup_ind=1, ref_ind=0, missing_value=-1,
        sorter=np.argsort(unsorted[:, 1]),
    )
    assert with_sorter[0] == 2


def test_the_error_names_the_first_out_of_order_entry():
    """So a bad reference file can be located, not just rejected."""
    lookup = np.column_stack([np.array([10, 20, 30, 25, 40]), np.arange(5)])

    with pytest.raises(ValueError) as excinfo:
        _matching_voxel_indices(np.array([30]), lookup)

    message = str(excinfo.value)
    assert "entry 3 (25)" in message
    assert "entry 2 (30)" in message


def test_a_sorter_that_does_not_sort_raises():
    """The ordering can come from the sorter, so the sorter is checked too."""
    lookup = np.column_stack([np.array([30, 10, 20]), np.arange(3)])

    with pytest.raises(ValueError, match="in `sorter` order"):
        _matching_voxel_indices(
            np.array([10]), lookup, sorter=np.arange(3))

    ok = _matching_voxel_indices(
        np.array([10]), lookup, sorter=np.argsort(lookup[:, 0]))
    assert ok[0] == 1


def test_equal_neighbouring_keys_are_ordered(lookup):
    """Monotonic means non-decreasing -- ties are legal and common."""
    tied = np.column_stack([np.array([10, 10, 20, 20, 20]), np.arange(5)])
    assert _matching_voxel_indices(np.array([20]), tied)[0] == 2


def test_a_single_row_lookup_is_ordered():
    single = np.array([[7, 70]])
    assert _matching_voxel_indices(np.array([7, 8]), single, missing_value=-1).tolist() == [70, -1]


def test_the_ordering_check_is_remembered_per_array():
    """Characterization of the cost trade-off, and a warning.

    Checking is a pass over the column -- ~30 ms for the real 61.9M-row table,
    against 0.04 ms for the lookup itself -- and
    ``angle.find_closest_streamline`` looks one voxel up per call. So the answer
    is cached per array, which means an array reordered *in place* after its
    first successful use is not re-checked. Reference files are read once and
    left alone, so this is a fair trade; do not "fix" it by dropping the cache
    without restoring the per-call cost.
    """
    lookup = np.column_stack([np.array([10, 20, 30]), np.array([1, 2, 3])])
    assert _matching_voxel_indices(np.array([20]), lookup)[0] == 2

    lookup[:, 0] = [30, 20, 10]
    _matching_voxel_indices(np.array([20]), lookup)  # no raise: already checked

    # a fresh array with the same contents is checked, and rejected
    with pytest.raises(ValueError, match="must increase monotonically"):
        _matching_voxel_indices(np.array([20]), lookup.copy())


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


def test_a_query_past_the_last_key_is_missing(lookup):
    """The binary search runs off the end of the table here.

    There is no row at the returned insertion point to compare against, so the
    "is it present?" answer has to come from somewhere other than an index.
    """
    result = _matching_voxel_indices(
        np.array([40, 41, 10_000]), lookup, missing_value=-1)
    assert np.array_equal(result, np.array([100, -1, -1]))


def test_an_empty_lookup_returns_all_missing():
    result = _matching_voxel_indices(
        np.array([1, 2, 3]), np.zeros((0, 2), dtype=int), missing_value=-1)
    assert np.array_equal(result, np.array([-1, -1, -1]))


@pytest.mark.parametrize("with_sorter", [False, True])
def test_the_key_column_is_never_scanned_end_to_end(lookup, monkeypatch, with_sorter):
    """The helper must not pay a cost proportional to the size of the lookup.

    It used to ask ``np.isin`` whether each query was present, which walks and
    internally sorts the whole key column -- 61.9 million rows for the real
    ``closest surface voxel`` table -- however few voxels were queried.
    ``angle.find_closest_streamline`` queries exactly one.

    Forbidding ``np.isin``/``np.in1d`` stands in for that O(rows) cost; a
    timing assertion would say the same thing far less reliably. Membership now
    comes from the ``np.searchsorted`` the helper already performs.
    """
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "membership must come from the binary search, not a full-column scan")

    monkeypatch.setattr(np, "isin", forbidden)
    monkeypatch.setattr(np, "in1d", forbidden, raising=False)  # gone in numpy 2

    sorter = np.argsort(lookup[:, 0], kind="stable") if with_sorter else None
    result = _matching_voxel_indices(
        np.array([10, 15, 40]), lookup, sorter=sorter, missing_value=-1)
    assert np.array_equal(result, np.array([700, -1, 100]))
