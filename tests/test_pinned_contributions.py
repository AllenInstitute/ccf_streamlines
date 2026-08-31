"""Tests pinning the two open contributions.

Each open pull request claims to fix a specific defect. The tests here assert
the *fixed* behaviour and are marked ``xfail(strict=True)``, so:

- on the default branch they xfail, the suite stays green, and CI remains a
  useful gate on unrelated work;
- merging the contribution turns each into a reported XPASS failure, which is
  machine-checkable evidence that the change does what it claims and the
  signal to delete the marker.

Pinned here:

- PR #13 / issue #14 -- ``project_coordinates`` crashes when
  ``drop_voxels_outside_view_streamlines`` is False and no voxel falls outside
  the view.
- PR #11 / issue #12 -- ``IsocortexCoordinateProjector.project_coordinates``
  results depend on CPU architecture, because tied view-lookup keys are
  resolved by an unstable sort.

**Sequencing note.** These two interact. The clean stable-sort test would
query a single in-view coordinate, which is exactly the condition that
triggers PR #13's empty-concatenation crash. The stable-sort test therefore
sets ``drop_voxels_outside_view_streamlines=True`` to sidestep that branch.
Do not "simplify" that away before #13 lands.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import (
    IsocortexCoordinateProjector,
    _matching_voxel_indices,
)

PR13 = "pinned to AllenInstitute/ccf_streamlines#13 (issue #14); remove this marker when it merges"
PR11 = "pinned to AllenInstitute/ccf_streamlines#11 (issue #12); remove this marker when it merges"


@pytest.fixture
def coordinate_projector(mini_ccf):
    return IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        projection_file=mini_ccf.view_lookup_file,
    )


# ---------------------------------------------------------------------------
# PR #13 -- empty nearest-streamline search
# ---------------------------------------------------------------------------


def test_project_coordinates_survives_an_empty_nearest_streamline_search(
    mini_ccf, coordinate_projector
):
    """Every coordinate already lies on a streamline the view uses.

    ``_calculate_2d_coordinates`` then has nothing to search for, builds no
    chunks, and calls ``np.concatenate([])``, which raises
    ``ValueError: need at least one array to concatenate``. Nothing about the
    request is invalid: the answer is simply that no coordinate needed
    snapping.
    """
    coords = np.vstack(
        [mini_ccf.coord_on_path(int(p), 3) for p in mini_ccf.in_view_path_indices[:3]]
    )

    result = coordinate_projector.project_coordinates(coords, hemisphere="left")

    # With nothing to snap, the answer must equal the drop-outside answer.
    expected = coordinate_projector.project_coordinates(
        coords, hemisphere="left", drop_voxels_outside_view_streamlines=True
    )
    assert np.array_equal(result, expected)
    assert not np.isnan(result).any()


def test_nearest_streamline_search_still_runs_when_there_is_work_to_do(
    mini_ccf, coordinate_projector
):
    """The complement of the case above, and it must keep working after #13.

    One coordinate sits on a streamline the view does not use, so it is
    snapped to the nearest streamline that the view does use.
    """
    in_view = mini_ccf.coord_on_path(int(mini_ccf.in_view_path_indices[0]), 3)
    out_of_view = mini_ccf.coord_on_path(int(mini_ccf.out_of_view_path_indices[0]), 3)
    coords = np.vstack([in_view, out_of_view])

    snapped = coordinate_projector.project_coordinates(coords, hemisphere="left")
    dropped = coordinate_projector.project_coordinates(
        coords, hemisphere="left", drop_voxels_outside_view_streamlines=True
    )

    # The in-view coordinate is unaffected by the choice.
    assert np.array_equal(snapped[0, :2], dropped[0, :2])
    # The out-of-view one is dropped in one mode and snapped in the other.
    assert np.isnan(dropped[1, :2]).all()
    assert not np.isnan(snapped[1, :2]).any()


# ---------------------------------------------------------------------------
# PR #11 -- stable sort for the view-lookup sorter
# ---------------------------------------------------------------------------


def _tie_reversing_argsort(real_argsort):
    """A sort that is legal but hostile: it sorts the keys and reverses ties.

    Delegating on an explicitly stable request is what makes this a *fix
    detector* rather than an oracle that passes either way -- with the fix in
    place the call asks for ``kind="stable"`` and gets the real sort.

    The permutation it returns is a valid sort of the keys, which it must be:
    other call sites use sorting for tie-order-independent sort-gather-unsort
    round trips, and those have to keep working while this is installed.
    """

    def tie_reversing_argsort(a, axis=-1, kind=None, order=None):
        if kind == "stable" or np.ndim(a) != 1:
            return real_argsort(a, axis=axis, kind=kind, order=order)
        return np.lexsort((-np.arange(len(a)), a))

    return tie_reversing_argsort


def _tie_preserving_argsort(real_argsort):
    """The mirror image: always resolve ties in row order.

    Used to give the comparison in the end-to-end test a fixed reference that
    does not depend on what the platform's default sort happens to do.
    """

    def tie_preserving_argsort(a, axis=-1, kind=None, order=None):
        if np.ndim(a) != 1:
            return real_argsort(a, axis=axis, kind=kind, order=order)
        return real_argsort(a, axis=axis, kind="stable", order=order)

    return tie_preserving_argsort


def test_matching_voxel_indices_resolves_a_tie_by_sorter_order(mini_ccf):
    """Characterization: this passes both before and after the fix.

    It documents *why* the caller must request stability. The helper resolves
    keys by ``np.searchsorted``'s left insertion point, so for a tied key it
    returns whichever tied entry the supplied sorter placed first. The helper
    is not wrong; the caller choosing an unstable sort is.
    """
    tied_key = mini_ccf.tied_view_keys()[0]
    rows = np.flatnonzero(mini_ccf.view_lookup[:, 1] == tied_key)
    assert len(rows) > 1
    view_indices = mini_ccf.view_lookup[rows, 0]

    query = np.array([tied_key])

    stable = _matching_voxel_indices(
        query,
        mini_ccf.view_lookup,
        lookup_ind=1,
        ref_ind=0,
        missing_value=-1,
        sorter=np.argsort(mini_ccf.view_lookup[:, 1], kind="stable"),
    )
    hostile = _matching_voxel_indices(
        query,
        mini_ccf.view_lookup,
        lookup_ind=1,
        ref_ind=0,
        missing_value=-1,
        sorter=np.lexsort((-np.arange(len(mini_ccf.view_lookup)), mini_ccf.view_lookup[:, 1])),
    )

    # Because view-lookup rows are in increasing view-index order within a tied
    # group, a stable sort yields the smallest tied view index. That is the
    # exact contract PR #11 makes the caller ask for.
    assert stable[0] == view_indices.min()
    assert hostile[0] == view_indices.max()
    assert stable[0] != hostile[0], "the fixture must have a genuine tie"


def test_the_hostile_sort_is_a_legal_sort(mini_ccf):
    """Otherwise the test above would prove nothing about sort *stability*."""
    keys = mini_ccf.view_lookup[:, 1]
    hostile = np.lexsort((-np.arange(len(keys)), keys))
    assert np.all(np.diff(keys[hostile]) >= 0)
    assert sorted(hostile.tolist()) == list(range(len(keys)))


def test_the_obvious_version_of_this_test_would_prove_nothing(mini_ccf):
    """Whether the default sort reorders these ties is a property of the build.

    A naive version of the end-to-end test below would call
    ``project_coordinates`` twice and compare, relying on numpy's default sort
    to reorder ties. Whether it does is not something a test can count on:

    - numpy falls back to insertion sort, which is stable, below roughly
      sixteen elements, so on many builds the default and stable sorts of
      these fourteen keys are identical and the naive test is vacuous;
    - but numpy also dispatches to architecture-specific SIMD kernels, and on
      GitHub's x86-64 runners the two *do* differ at this size, while on the
      arm64 runners and on a local x86-64 workstation they do not.

    So the naive test is either vacuous or platform-dependent, and which one
    varies by machine. That is exactly the situation the injected sorts below
    exist to remove. This test asserts only what is true on every build: both
    orderings are valid sorts of the keys, and they can legitimately disagree
    about ties.
    """
    keys = mini_ccf.view_lookup[:, 1]
    default = np.argsort(keys)
    stable = np.argsort(keys, kind="stable")

    assert np.all(np.diff(keys[default]) >= 0)
    assert np.all(np.diff(keys[stable]) >= 0)
    assert sorted(default.tolist()) == sorted(stable.tolist())


@pytest.mark.xfail(strict=True, reason=PR11)
def test_2d_coordinates_are_independent_of_sort_tie_order(
    mini_ccf, coordinate_projector, monkeypatch
):
    """End-to-end: the projected pixel must not depend on how ties are ordered.

    The defect exists only as a divergence between processor architectures --
    numpy dispatches its default quicksort to architecture-specific SIMD
    kernels, which order ties differently. A single-architecture comparison
    cannot catch it, so a legal-but-hostile tie order stands in for the other
    architecture.

    ``drop_voxels_outside_view_streamlines=True`` sidesteps the PR #13 crash;
    see the module docstring.
    """
    # A coordinate on a streamline that two view pixels both reference.
    tied_key = mini_ccf.tied_view_keys()[0]
    path_index = int(np.flatnonzero(mini_ccf.paths[:, 0] == tied_key)[0])
    pixels = mini_ccf.view_pixels_for_path(path_index)
    assert len(pixels) > 1

    coords = mini_ccf.coord_on_path(path_index, 3).reshape(1, 3)
    kwargs = dict(hemisphere="left", drop_voxels_outside_view_streamlines=True)

    real_argsort = np.argsort

    # Both references are injected, so neither depends on what this platform's
    # default sort happens to do with ties -- which varies by numpy build.
    monkeypatch.setattr(np, "argsort", _tie_preserving_argsort(real_argsort))
    preserved = coordinate_projector.project_coordinates(coords, **kwargs)

    monkeypatch.setattr(np, "argsort", _tie_reversing_argsort(real_argsort))
    reversed_ties = coordinate_projector.project_coordinates(coords, **kwargs)

    assert np.array_equal(preserved, reversed_ties), (
        "the projected coordinate changed when tied view-lookup keys were "
        "ordered differently, which is what differs between architectures"
    )
    # And the answer is the smallest tied view index, not merely a stable one.
    expected_row, expected_col = min(pixels)
    assert (reversed_ties[0, 0], reversed_ties[0, 1]) == (expected_row, expected_col)


def test_sort_gather_unsort_round_trips_survive_a_hostile_tie_order(
    mini_ccf, coordinate_projector, monkeypatch
):
    """The other sorting call sites are tie-order-independent, and must stay so.

    ``_path_lookup_chunked`` sorts, gathers, then unsorts. Any legal
    permutation must round-trip. If this breaks, the hostile sort above is
    testing the wrong thing.
    """
    coords = np.vstack(
        [mini_ccf.coord_on_path(int(p), 2) for p in mini_ccf.in_view_path_indices[:4]]
    )
    baseline = coordinate_projector.project_depths(coords)

    monkeypatch.setattr(np, "argsort", _tie_reversing_argsort(np.argsort))
    hostile = coordinate_projector.project_depths(coords)

    assert np.array_equal(baseline, hostile)
