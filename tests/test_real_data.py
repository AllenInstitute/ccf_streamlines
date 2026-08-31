"""The opt-in real-data tier.

Skipped unless ``CCF_STREAMLINES_TEST_DATA`` names the directory holding the
real assets, in the published layout::

    <dir>/streamlines/surface_paths_10_v3.h5
    <dir>/streamlines/closest_surface_voxel_lookup.h5
    <dir>/view_lookup/*.h5
    <dir>/cortical_metrics/cortical_layers_10_v2.h5
    <dir>/master_updated/*.nrrd, labelDescription_ITKSNAPColor.txt

It never runs in continuous integration. Its purpose is drift detection: the
mini-CCF fixtures encode a set of structural claims about these files, and a
synthetic-only suite invites those claims quietly becoming self-consistent
fiction -- exactly as ``data_file_info.md`` already did. Every assertion here
restates a property the fixtures depend on.

It also confirms the tie-density and sort-stability behaviour at real scale,
which is a property no miniature fixture reproduces.

Run with::

    CCF_STREAMLINES_TEST_DATA=/path/to/handoff uv run pytest -m real_data
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from tests.mini_ccf import LAYER_KEYS, reference_layer_thicknesses

pytestmark = pytest.mark.real_data

VIEWS_WITH_TIES = ("flatmap_butterfly", "flatmap_dorsal", "rotated")
VIEWS_WITHOUT_TIES = ("back", "bottom", "front", "medial", "side", "top")


def _asset(real_data_dir, *parts):
    path = real_data_dir.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"{path} not present")
    return path


@pytest.fixture(scope="session")
def surface_paths(real_data_dir):
    return _asset(real_data_dir, "streamlines", "surface_paths_10_v3.h5")


@pytest.fixture(scope="session")
def closest_surface_voxels(real_data_dir):
    return _asset(real_data_dir, "streamlines", "closest_surface_voxel_lookup.h5")


@pytest.fixture(scope="session")
def layer_thickness_file(real_data_dir):
    return _asset(real_data_dir, "cortical_metrics", "cortical_layers_10_v2.h5")


# -- surface paths file ----------------------------------------------------


def test_surface_paths_has_no_file_level_attributes(surface_paths):
    """The fixture writes none. The superseded generation had origin/size/spacing."""
    with h5py.File(surface_paths, "r") as f:
        assert dict(f.attrs) == {}


def test_surface_paths_has_only_the_flattened_lookup(surface_paths):
    with h5py.File(surface_paths, "r") as f:
        assert "paths" in f
        assert "volume lookup flat" in f
        assert "volume lookup" not in f, (
            "a 3-D volume lookup would mean project_path_ordered_data works "
            "again and its pinned test should be revisited"
        )


def test_the_flat_lookup_records_the_original_shape(surface_paths):
    """The only attribute in the file, and how volume shape is recovered."""
    with h5py.File(surface_paths, "r") as f:
        dset = f["volume lookup flat"]
        assert list(dset.attrs) == ["original shape"]
        shape = tuple(int(v) for v in dset.attrs["original shape"])

    assert shape == (1320, 800, 1140)
    assert len(set(shape)) == 3, "the fixture relies on three distinct axes"


def test_the_flat_lookup_is_populated_only_at_streamline_starts(surface_paths):
    """The central fixture claim: ~0.12% fill, sentinel elsewhere.

    Checked on a sample of streamlines rather than the whole 1.2-billion-cell
    array, which would take minutes to read.
    """
    with h5py.File(surface_paths, "r") as f:
        paths = f["paths"]
        flat = f["volume lookup flat"]
        n_paths = paths.shape[0]

        for path_index in (0, 1, 500, n_paths // 2, n_paths - 1):
            row = paths[path_index, :]
            row = row[row > 0]
            assert flat[int(row[0])] == path_index

            later = np.unique(row[1:]).astype(np.int64)
            assert np.all(flat[later.tolist()] == -1), (
                "a non-start voxel is populated; the fixture fills start "
                "voxels only"
            )


def test_the_flat_lookup_sentinel_is_minus_one(surface_paths):
    with h5py.File(surface_paths, "r") as f:
        sample = f["volume lookup flat"][:1_000_000]
    assert sample.min() == -1
    # Overwhelmingly sentinel: fill is a fraction of a percent.
    assert np.count_nonzero(sample != -1) < 0.01 * sample.size


def test_paths_are_zero_padded_on_the_right(surface_paths):
    with h5py.File(surface_paths, "r") as f:
        rows = f["paths"][:100, :]

    for row in rows:
        nonzero = np.flatnonzero(row)
        if nonzero.size:
            # every entry up to the last non-zero is non-zero: no interior gaps
            assert nonzero[-1] - nonzero[0] + 1 == nonzero.size
            assert nonzero[0] == 0


def test_no_streamline_starts_at_flat_index_zero(surface_paths):
    """Zero doubles as the padding value and the max/min projection sentinel."""
    with h5py.File(surface_paths, "r") as f:
        starts = f["paths"][:, 0]
    assert starts.min() > 0


# -- closest surface voxel lookup ------------------------------------------


def test_closest_surface_voxel_is_a_two_column_sorted_table(closest_surface_voxels):
    """`_matching_voxel_indices` searchsorts column 0 with no sorter."""
    with h5py.File(closest_surface_voxels, "r") as f:
        dset = f["closest surface voxel"]
        assert dset.ndim == 2 and dset.shape[1] == 2
        head = dset[:2_000_000]

    assert np.all(np.diff(head[:, 0].astype(np.int64)) > 0)


def test_closest_surface_voxel_targets_are_streamline_starts(
    closest_surface_voxels, surface_paths
):
    with h5py.File(surface_paths, "r") as f:
        starts = set(f["paths"][:, 0].tolist())
    with h5py.File(closest_surface_voxels, "r") as f:
        sample = f["closest surface voxel"][:200_000:997, 1]

    assert set(sample.tolist()) <= starts


# -- layer thickness file --------------------------------------------------


def test_layer_datasets_are_present_and_two_three_is_nested(layer_thickness_file):
    with h5py.File(layer_thickness_file, "r") as f:
        for key in LAYER_KEYS:
            assert key in f, f"{key} missing"
        assert isinstance(f["Isocortex layer 2"], h5py.Group)
        assert isinstance(f["Isocortex layer 2/3"], h5py.Dataset)


def test_layer_rows_are_start_end_thickness_and_contiguous(layer_thickness_file):
    with h5py.File(layer_thickness_file, "r") as f:
        rows = {k: f[k][:5000, :] for k in LAYER_KEYS}

    assert all(v.shape[1] == 3 for v in rows.values())

    for i in range(0, 5000, 313):
        previous_end = 0.0
        for key in LAYER_KEYS:
            start, end, thickness = rows[key][i, :]
            if start == 0 and end == 0:
                continue  # layer absent from this streamline
            assert end - start == pytest.approx(thickness, abs=1e-2)
            assert start == pytest.approx(previous_end, abs=1e-2)
            previous_end = end


def test_layer_totals_equal_arc_length_plus_one_voxel(
    layer_thickness_file, surface_paths
):
    """The relation the fixtures scale their layer thicknesses to.

    Half a voxel of pia plus half of white matter. If this drifts, every
    layer-normalized depth in the fixtures becomes untrustworthy.
    """
    resolution = np.array([10, 10, 10])
    with h5py.File(surface_paths, "r") as f:
        shape = tuple(int(v) for v in f["volume lookup flat"].attrs["original shape"])
        paths = f["paths"][:5000, :]
    with h5py.File(layer_thickness_file, "r") as f:
        totals = np.sum([f[k][:5000, 2] for k in LAYER_KEYS], axis=0)

    checked = 0
    for i in range(0, 5000, 197):
        row = paths[i, :]
        row = row[row > 0]
        if row.size < 3 or totals[i] <= 0:
            continue
        xyz = np.array(np.unravel_index(row, shape)).T * resolution
        arc = np.sqrt((np.diff(xyz, axis=0) ** 2).sum(axis=1)).sum()
        assert totals[i] - arc == pytest.approx(float(np.mean(resolution)), abs=0.5)
        checked += 1

    assert checked > 10


# -- view lookups ----------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS_WITH_TIES + VIEWS_WITHOUT_TIES)
def test_view_lookup_attribute_types(real_data_dir, view):
    """Origin and spacing are byte strings; size and view size are integers.

    The fixture reproduces this asymmetry because the loading code decodes
    spacing -- a fixture writing it as integers would pass while the real file
    crashed.
    """
    path = _asset(real_data_dir, "view_lookup", f"{view}.h5")
    with h5py.File(path, "r") as f:
        assert all(isinstance(d, bytes) for d in f.attrs["spacing"])
        assert all(isinstance(d, bytes) for d in f.attrs["origin"])
        assert np.issubdtype(f.attrs["size"].dtype, np.integer)
        assert np.issubdtype(f.attrs["view size"].dtype, np.integer)
        assert f["view lookup"].shape[1] == 2


@pytest.mark.parametrize("view", VIEWS_WITH_TIES)
def test_the_flatmap_and_rotated_views_have_tied_keys(real_data_dir, view):
    """The scope of issue #12, and the property the fixture reproduces."""
    path = _asset(real_data_dir, "view_lookup", f"{view}.h5")
    with h5py.File(path, "r") as f:
        keys = f["view lookup"][:, 1]

    _, counts = np.unique(keys, return_counts=True)
    assert np.count_nonzero(counts > 1) > 0
    assert counts.max() > 1


@pytest.mark.parametrize("view", VIEWS_WITHOUT_TIES)
def test_the_single_projection_views_have_no_tied_keys(real_data_dir, view):
    """So the architecture-dependence cannot arise for these views at all."""
    path = _asset(real_data_dir, "view_lookup", f"{view}.h5")
    with h5py.File(path, "r") as f:
        keys = f["view lookup"][:, 1]

    assert np.unique(keys).size == keys.size


@pytest.mark.parametrize("view", VIEWS_WITH_TIES)
def test_tied_rows_are_in_increasing_view_index_order(real_data_dir, view):
    """The contract that makes "stable" the right answer rather than merely a
    deterministic one: a stable sort resolves a tie to the smallest view index."""
    path = _asset(real_data_dir, "view_lookup", f"{view}.h5")
    with h5py.File(path, "r") as f:
        lookup = f["view lookup"][:]

    ordered = lookup[np.argsort(lookup[:, 1], kind="stable")]
    starts = np.flatnonzero(np.r_[True, np.diff(ordered[:, 1]) != 0])
    ends = np.r_[starts[1:], len(ordered)]

    violations = 0
    for a, b in zip(starts, ends):
        if b - a > 1 and not np.all(np.diff(ordered[a:b, 0].astype(np.int64)) > 0):
            violations += 1
    assert violations == 0


def test_the_default_sort_reorders_ties_at_real_scale(real_data_dir):
    """The defect, at genuine tie density.

    numpy dispatches its default quicksort to architecture-specific SIMD
    kernels above a size threshold, so which tied row lands first differs
    between architectures. Skipped where the platform's default sort happens
    not to reorder, so it demonstrates the defect where it bites without being
    flaky elsewhere.
    """
    path = _asset(real_data_dir, "view_lookup", "flatmap_butterfly.h5")
    with h5py.File(path, "r") as f:
        keys = f["view lookup"][:, 1]

    default = np.argsort(keys)
    stable = np.argsort(keys, kind="stable")

    if np.array_equal(default, stable):
        pytest.skip(
            "this platform's default sort happens to be stable for these keys; "
            "the divergence is architecture-specific"
        )

    # Both are legal sorts...
    assert np.all(np.diff(keys[default]) >= 0)
    assert np.all(np.diff(keys[stable]) >= 0)
    # ...but they disagree about ties, which is exactly what changes results.
    assert not np.array_equal(default, stable)


# -- end to end ------------------------------------------------------------


def test_a_real_projection_runs_through_the_public_interface(
    real_data_dir, surface_paths
):
    """The one heavy test: constructs a projector from the real files.

    Reads the whole 1.1 GB paths array, so it is slow and memory-hungry. It is
    here because a fixture-only suite cannot tell you that the loading code
    still works against the files users actually download.
    """
    from ccf_streamlines.projection import Isocortex2dProjector

    view = _asset(real_data_dir, "view_lookup", "top.h5")
    projector = Isocortex2dProjector(str(view), str(surface_paths), hemisphere="left")

    assert projector.resolution == (10, 10, 10)
    assert tuple(projector.volume_shape) == (1320, 800, 1140)
    assert projector.paths.shape[0] == projector.view_lookup.shape[0]

    volume = np.zeros(projector.volume_shape, dtype=np.uint8)
    lit_path = projector.paths[0, :]
    lit_path = lit_path[lit_path > 0]
    volume.flat[lit_path[3]] = 200

    result = projector.project_volume(volume, kind="max")

    assert result.shape == tuple(projector.view_size)
    row, col = np.unravel_index(int(projector.view_lookup[0, 0]), projector.view_size)
    assert result[row, col] == 200


def test_mirrored_queries_find_mirrored_streamlines(surface_paths, closest_surface_voxels):
    """`find_closest_streamline` on real streamlines, which actually differ.

    The mini-CCF's streamlines all run straight down +y and are identical from
    one lateral position to the next, so an off-by-one in the hemisphere
    reflection returns a neighbouring streamline that *looks* the same. Only
    real streamlines, which curve differently at every position, can show that
    the wrong one came back -- under the superseded `z_size - z` two thirds of
    the voxels sampled here returned a streamline of a different length
    (issue #33).
    """
    import h5py as _h5py

    from ccf_streamlines.angle import find_closest_streamline

    shape = (1320, 800, 1140)
    z_size = shape[2]
    resolution = (10, 10, 10)

    with _h5py.File(surface_paths, "r") as f:
        sample = f["paths"][::30000, :]
    with _h5py.File(closest_surface_voxels, "r") as f:
        reference = f["closest surface voxel"][:]

    # Voxels partway down a streamline, on the left, where the reference lives.
    left_voxels = []
    for row in sample:
        valid = row[row > 0]
        if len(valid) > 5:
            voxel = np.unravel_index(valid[len(valid) // 2], shape)
            if voxel[2] < z_size // 2:
                left_voxels.append(voxel)
    # Three is enough: each call scans the whole 62M-row reference through
    # `_matching_voxel_indices`, so they cost seconds apiece.
    left_voxels = left_voxels[:3]
    assert left_voxels, "no left-hemisphere in-cortex voxels sampled"

    for voxel in left_voxels:
        left = (np.array(voxel) + 0.5) * np.array(resolution, dtype=float)
        right = left.copy()
        right[2] = z_size * resolution[2] - left[2]

        from_left = find_closest_streamline(
            left, reference, str(surface_paths),
            resolution=resolution, volume_shape=shape,
        )
        from_right = find_closest_streamline(
            right, reference, str(surface_paths),
            resolution=resolution, volume_shape=shape,
        )

        assert from_left.size > 0, f"the left-hand control must be in cortex ({voxel})"
        mirrored = from_left.copy()
        mirrored[:, 2] = (z_size - 1) * resolution[2] - from_left[:, 2]
        assert from_right.shape == mirrored.shape, f"different streamline ({voxel})"
        assert np.array_equal(from_right, mirrored), f"{voxel}"
