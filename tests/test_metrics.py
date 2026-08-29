"""Layer thickness measurement, against streamlines whose layers we chose.

The layer memberships are assigned voxel by voxel here, so the returned start,
end and thickness are all derivable by hand: a straight path of unit-spaced
voxels at 10 micron resolution puts layer *i* between 10*i and 10*(i+1).
"""

import numpy as np
import pytest

from ccf_streamlines.metrics import measure_streamline_layer_thicknesses

# Structure set IDs from the mouse ontology, in cortical depth order. These are
# also the values `measure_streamline_layer_thicknesses` looks for, and their
# numeric order matches their depth order -- which the function relies on when
# it sorts annotations to stop layers being intercalated.
LAYER_IDS = [667481440, 667481441, 667481445, 667481446, 667481449, 667481450]
LAYER_NAMES = [
    "Isocortex layer 1",
    "Isocortex layer 2/3",
    "Isocortex layer 4",
    "Isocortex layer 5",
    "Isocortex layer 6a",
    "Isocortex layer 6b",
]
RESOLUTION = (10, 10, 10)
SHAPE = (4, 10, 3)


def _straight_path(x, z, layer_ids, padded_length=8):
    """A path down +y at (x, z), one voxel per entry of ``layer_ids``.

    Returns the padded flat-index row and a volume labelled to match.
    """
    volume = np.zeros(SHAPE, dtype=np.int64)
    ys = np.arange(1, 1 + len(layer_ids))
    for y, layer in zip(ys, layer_ids):
        volume[x, y, z] = layer
    flat = np.ravel_multi_index((np.full_like(ys, x), ys, np.full_like(ys, z)), SHAPE)
    row = np.zeros(padded_length, dtype=np.int64)
    row[: len(flat)] = flat
    return row, volume


def test_one_voxel_per_layer_gives_ten_micron_layers():
    """Six voxels, one per layer, so every layer is exactly one voxel thick."""
    row, volume = _straight_path(1, 1, LAYER_IDS)
    paths = row.reshape(1, -1)

    result = measure_streamline_layer_thicknesses(volume, paths, RESOLUTION)

    assert set(result) == set(LAYER_NAMES)
    for i, name in enumerate(LAYER_NAMES):
        start, end, thickness = result[name][0]
        assert start == pytest.approx(10.0 * i)
        assert end == pytest.approx(10.0 * (i + 1))
        assert thickness == pytest.approx(10.0)


def test_thicknesses_follow_the_number_of_voxels_in_each_layer():
    """Layer 2/3 given three voxels must come back three times as thick."""
    layers = [
        LAYER_IDS[0],
        LAYER_IDS[1], LAYER_IDS[1], LAYER_IDS[1],
        LAYER_IDS[2],
        LAYER_IDS[3],
        LAYER_IDS[4],
        LAYER_IDS[5],
    ]
    row, volume = _straight_path(1, 1, layers, padded_length=10)

    result = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), RESOLUTION)

    assert result["Isocortex layer 1"][0][2] == pytest.approx(10.0)
    assert result["Isocortex layer 2/3"][0][2] == pytest.approx(30.0)
    assert result["Isocortex layer 2/3"][0][0] == pytest.approx(10.0)
    assert result["Isocortex layer 2/3"][0][1] == pytest.approx(40.0)
    assert result["Isocortex layer 4"][0][0] == pytest.approx(40.0)


def test_an_absent_layer_reports_all_zeros():
    """This is how the projectors detect that a layer is missing."""
    layers = [i for i in LAYER_IDS if i != LAYER_IDS[2]]  # no layer 4
    row, volume = _straight_path(1, 1, layers)

    result = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), RESOLUTION)

    assert np.array_equal(result["Isocortex layer 4"][0], np.zeros(3))
    # ...and the layers around it stay contiguous
    assert result["Isocortex layer 2/3"][0][1] == pytest.approx(20.0)
    assert result["Isocortex layer 5"][0][0] == pytest.approx(20.0)


def test_total_thickness_equals_the_number_of_annotated_voxels():
    row, volume = _straight_path(1, 1, LAYER_IDS)
    result = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), RESOLUTION)

    total = sum(result[name][0][2] for name in LAYER_NAMES)
    assert total == pytest.approx(60.0)


def test_several_streamlines_are_measured_independently():
    layers_a = LAYER_IDS
    layers_b = [LAYER_IDS[0], LAYER_IDS[0], LAYER_IDS[1], LAYER_IDS[2],
                LAYER_IDS[3], LAYER_IDS[4], LAYER_IDS[5]]
    row_a, volume = _straight_path(1, 1, layers_a, padded_length=10)
    row_b, volume_b = _straight_path(2, 2, layers_b, padded_length=10)
    volume = volume + volume_b

    paths = np.vstack([row_a, row_b])
    result = measure_streamline_layer_thicknesses(volume, paths, RESOLUTION)

    assert result["Isocortex layer 1"][0][2] == pytest.approx(10.0)
    assert result["Isocortex layer 1"][1][2] == pytest.approx(20.0)


def test_duplicate_consecutive_voxels_are_collapsed_before_measuring():
    """A path that lingers on a voxel must not double-count its layer."""
    row, volume = _straight_path(1, 1, LAYER_IDS, padded_length=10)
    with_dupes = np.zeros(12, dtype=np.int64)
    values = row[row > 0]
    doubled = np.repeat(values[:1], 2)  # visit the first voxel twice
    packed = np.concatenate([doubled, values[1:]])
    with_dupes[: len(packed)] = packed

    plain = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), RESOLUTION)
    duped = measure_streamline_layer_thicknesses(
        volume, with_dupes.reshape(1, -1), RESOLUTION
    )

    for name in LAYER_NAMES:
        assert plain[name][0] == pytest.approx(duped[name][0])


def test_resolution_scales_the_measured_thicknesses():
    row, volume = _straight_path(1, 1, LAYER_IDS)

    at_10 = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), (10, 10, 10))
    at_20 = measure_streamline_layer_thicknesses(volume, row.reshape(1, -1), (20, 20, 20))

    for name in LAYER_NAMES:
        assert at_20[name][0][2] == pytest.approx(2 * at_10[name][0][2])
