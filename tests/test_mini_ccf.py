"""Tests of the fixture factory itself.

If the factory is wrong, every test built on it is wrong in the same direction
and none of them notice. These check that the structural properties the rest of
the suite relies on actually hold, and that an invalid override fails with a
readable message rather than deep inside library code.
"""

import numpy as np
import pytest

from tests import mini_ccf as mc


def test_files_are_small_enough_to_build_per_test(mini_ccf):
    """The whole atlas is tens of KB, so a fresh one per test costs nothing."""
    total = sum(p.stat().st_size for p in mini_ccf.root.iterdir())
    assert total < 100_000


def test_volume_axes_are_mutually_distinct_and_ordered_like_the_real_atlas(mini_ccf):
    # The real volume is (1320, 800, 1140): x > z > y. A cube would silently
    # pass an axis-transposition bug.
    x, y, z = mini_ccf.volume_shape
    assert len({x, y, z}) == 3
    assert x > z > y


def test_lateral_axis_is_even_and_streamlines_sit_on_one_side(mini_ccf):
    z_size = mini_ccf.volume_shape[2]
    assert z_size % 2 == 0, "midline must fall on a voxel boundary"
    assert np.all(mini_ccf.path_starts[:, 2] < z_size / 2)
    # Reflection maps one half onto the other exactly.
    reflected = z_size - mini_ccf.path_starts[:, 2]
    assert np.all(reflected > z_size / 2)


def test_at_least_three_lateral_positions(mini_ccf):
    # Fewer than three makes a nearest-streamline search degenerate.
    assert len(set(mini_ccf.path_starts[:, 2].tolist())) >= 3


def test_planes_at_both_ends_of_the_dorsoventral_axis_are_outside_cortex(mini_ccf):
    voxels = np.vstack(
        [mini_ccf.path_voxels(i) for i in range(mini_ccf.paths.shape[0])]
    )
    y_size = mini_ccf.volume_shape[1]
    assert voxels[:, 1].min() >= 1
    assert voxels[:, 1].max() <= y_size - 2


def test_no_streamline_voxel_has_flat_index_zero(mini_ccf):
    """Zero is both the padding value and the cell max/min projection clobbers."""
    used = mini_ccf.paths[mini_ccf.paths > 0]
    assert used.min() > 0
    assert 0 not in mini_ccf.paths[:, 0].tolist()


def test_flat_lookup_is_populated_only_at_streamline_starts(mini_ccf):
    """The real file has ~0.12% fill: start voxels only, sentinel elsewhere.

    This is sufficient because every call site queries only surface voxels,
    having first resolved an arbitrary voxel through the closest-surface-voxel
    lookup.
    """
    import h5py

    with h5py.File(mini_ccf.surface_paths_file, "r") as f:
        flat = f["volume lookup flat"][:]
        assert (
            tuple(f["volume lookup flat"].attrs["original shape"])
            == mini_ccf.volume_shape
        )

    for i in range(mini_ccf.paths.shape[0]):
        row = mini_ccf.paths[i, :]
        row = row[row > 0]
        assert flat[int(row[0])] == i
        # every subsequent voxel on the same streamline returns the sentinel
        assert np.all(flat[row[1:].astype(np.int64)] == -1)

    assert np.count_nonzero(flat != -1) == mini_ccf.paths.shape[0]


def test_surface_paths_file_has_no_file_level_attributes(mini_ccf):
    """The current-generation file carries none, unlike the superseded one."""
    import h5py

    with h5py.File(mini_ccf.surface_paths_file, "r") as f:
        assert dict(f.attrs) == {}
        assert "volume lookup" not in f, (
            "the current-generation file has only the flattened lookup"
        )


def test_view_lookup_encodes_spacing_as_bytes_and_sizes_as_integers(mini_ccf):
    """An asymmetry the loading code depends on.

    ``Isocortex2dProjector`` calls ``d.decode()`` on each spacing entry, so a
    fixture writing spacing as integers would pass while the real file crashed.
    """
    import h5py

    with h5py.File(mini_ccf.view_lookup_file, "r") as f:
        assert all(isinstance(d, bytes) for d in f.attrs["spacing"])
        assert all(isinstance(d, bytes) for d in f.attrs["origin"])
        assert np.issubdtype(f.attrs["size"].dtype, np.integer)
        assert np.issubdtype(f.attrs["view size"].dtype, np.integer)


def test_layer_key_with_a_slash_is_a_nested_group(mini_ccf):
    import h5py

    with h5py.File(mini_ccf.layer_thickness_file, "r") as f:
        assert isinstance(f["Isocortex layer 2"], h5py.Group)
        assert isinstance(f["Isocortex layer 2/3"], h5py.Dataset)


def test_layer_thicknesses_sum_to_arc_length_plus_one_voxel(mini_ccf):
    """Measured on the real assets: half a voxel of pia plus half of white matter.

    A fixture that ignores this pushes every depth query past the final layer
    boundary, so layer-normalized depths all come back as the missing sentinel.
    """
    one_voxel = float(np.mean(mini_ccf.resolution))
    for i in range(mini_ccf.paths.shape[0]):
        total = sum(
            float(mini_ccf.path_layer_thickness[k][i, 2]) for k in mc.LAYER_KEYS
        )
        assert total == pytest.approx(mini_ccf.path_arc_length(i) + one_voxel, abs=1e-2)


def test_one_streamline_has_a_deliberately_absent_layer(mini_ccf):
    i = mini_ccf.absent_layer_path_index
    k = mini_ccf.absent_layer_key
    assert tuple(mini_ccf.path_layer_thickness[k][i, :]) == (0, 0, 0)
    # ...and every other streamline has it
    others = [j for j in range(mini_ccf.paths.shape[0]) if j != i]
    assert all(mini_ccf.path_layer_thickness[k][j, 2] > 0 for j in others)


def test_view_lookup_has_tied_keys(mini_ccf):
    """Ties are what make the stable-sort defect reachable.

    Verified on the real assets: the flatmap and rotated views have tied keys
    (up to 11 view pixels per surface voxel); the six single-projection views
    have none at all.
    """
    tied = mini_ccf.tied_view_keys()
    assert len(tied) > 0
    # Within a tied group, the earlier row carries the smaller view index --
    # the contract a stable sort preserves.
    order = np.argsort(mini_ccf.view_lookup[:, 1], kind="stable")
    s = mini_ccf.view_lookup[order]
    for key in tied:
        rows = s[s[:, 1] == key]
        assert np.all(np.diff(rows[:, 0].astype(np.int64)) > 0)


def test_view_lookup_volume_column_is_not_sorted(mini_ccf):
    """As in the real files, which is why a sorter must be supplied."""
    assert not np.all(np.diff(mini_ccf.view_lookup[:, 1]) >= 0)


def test_view_content_fits_in_the_retained_half(mini_ccf):
    rows = np.unravel_index(mini_ccf.view_lookup[:, 0], mini_ccf.view_size)[0]
    assert rows.max() < mini_ccf.view_size[0] // 2
    assert mini_ccf.view_size[0] % 2 == 0
    assert mini_ccf.view_size[0] != mini_ccf.view_size[1]


def test_some_streamlines_are_absent_from_the_view(mini_ccf):
    """Without these the nearest-streamline fallback is unreachable."""
    assert len(mini_ccf.out_of_view_path_indices) > 0
    assert len(mini_ccf.in_view_path_indices) > 0


def test_closest_surface_voxel_targets_are_streamline_starts(mini_ccf):
    import h5py

    with h5py.File(mini_ccf.closest_surface_voxel_file, "r") as f:
        closest = f["closest surface voxel"][:]

    starts = set(mini_ccf.paths[:, 0].tolist())
    assert set(np.unique(closest[:, 1]).tolist()) <= starts
    # `_matching_voxel_indices` searchsorts this column with no sorter.
    assert np.all(np.diff(closest[:, 0].astype(np.int64)) > 0)


def test_volume_helper_returns_a_fresh_array_every_call(mini_ccf):
    """`project_volume` writes a sentinel into `volume.flat[0]` and never
    restores it, so tests that share one volume leak state into each other."""
    a = mini_ccf.volume()
    a.flat[0] = 99
    b = mini_ccf.volume()
    assert b.flat[0] == 0
    assert a is not b


def test_coord_on_path_lands_on_the_streamline(mini_ccf):
    from ccf_streamlines.coordinates import coordinates_to_voxels

    for step in range(mini_ccf.path_length):
        coord = mini_ccf.coord_on_path(0, step).reshape(1, 3)
        voxel = coordinates_to_voxels(coord, mini_ccf.resolution)[0]
        assert tuple(voxel) == tuple(mini_ccf.path_voxels(0)[step])


# -- parameterisation: one structural property at a time --------------------


def test_factory_can_vary_padded_length(mini_ccf_factory):
    mini = mini_ccf_factory(padded_length=16)
    assert mini.paths.shape[1] == 16


@pytest.mark.parametrize("padded_length", [9, 10, 13, 14, 15, 17, 19])
def test_padded_lengths_that_do_not_sum_are_rejected_readably(
    mini_ccf_factory, padded_length
):
    """Swept against the real layer ratios; these reshape-error deep in library code."""
    with pytest.raises(ValueError, match="sum to"):
        mini_ccf_factory(padded_length=padded_length)


@pytest.mark.parametrize("padded_length", [6, 7, 8])
def test_padded_lengths_that_drop_a_layer_are_rejected_readably(
    mini_ccf_factory, padded_length
):
    with pytest.raises(ValueError, match="zero voxels"):
        mini_ccf_factory(padded_length=padded_length)


@pytest.mark.parametrize("padded_length", [11, 12, 16, 18, 20])
def test_padded_lengths_that_work(mini_ccf_factory, padded_length):
    mini = mini_ccf_factory(padded_length=padded_length)
    blocks = mc._check_padded_length(padded_length, mini.layer_thicknesses)
    assert sum(blocks.values()) == padded_length
    assert all(v > 0 for v in blocks.values())


def test_eleven_is_the_true_minimum_with_all_six_layers(mini_ccf_factory):
    blocks = mc._check_padded_length(11, mc.reference_layer_thicknesses())
    assert list(blocks.values()) == [1, 3, 1, 3, 2, 1]
    # Twelve is the default because it gives layer 4 two voxels, not one,
    # which discriminates better.
    blocks12 = mc._check_padded_length(12, mc.reference_layer_thicknesses())
    assert list(blocks12.values()) == [1, 3, 2, 3, 2, 1]


def test_cubic_volume_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="mutually distinct"):
        mini_ccf_factory(volume_shape=(12, 12, 12))


def test_odd_lateral_axis_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="must be even"):
        mini_ccf_factory(volume_shape=(14, 10, 11))


def test_streamlines_crossing_the_midline_are_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="strictly on one side"):
        mini_ccf_factory(z_positions=(1, 2, 9))


def test_too_few_lateral_positions_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="three distinct lateral positions"):
        mini_ccf_factory(z_positions=(1, 2))


def test_streamlines_reaching_the_volume_edge_are_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="outside cortex"):
        mini_ccf_factory(y_first=0)


def test_a_view_with_no_ties_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="no tied keys"):
        mini_ccf_factory(n_tied_view_rows=0)


def test_odd_view_first_dimension_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="must be even"):
        mini_ccf_factory(view_size=(7, 5))


def test_square_view_is_rejected(mini_ccf_factory):
    with pytest.raises(ValueError, match="dimensions must differ"):
        mini_ccf_factory(view_size=(8, 8))
