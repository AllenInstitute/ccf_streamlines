"""``Isocortex3dProjector``: slab geometry and the three thickness types."""

import numpy as np
import pytest

from ccf_streamlines.projection import Isocortex3dProjector


def _projector(mini_ccf, **kwargs):
    kwargs.setdefault("hemisphere", "left")
    return Isocortex3dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, **kwargs
    )


@pytest.fixture
def unnormalized(mini_ccf):
    return _projector(mini_ccf)


@pytest.fixture
def normalized_full(mini_ccf):
    return _projector(mini_ccf, thickness_type="normalized_full")


@pytest.fixture
def normalized_layers(mini_ccf):
    return _projector(
        mini_ccf,
        thickness_type="normalized_layers",
        layer_thicknesses=mini_ccf.layer_thicknesses,
        streamline_layer_thickness_file=mini_ccf.layer_thickness_file,
    )


def _ramp(mini_ccf, path_index):
    """1..n down one streamline, so a depth profile is readable at a glance."""
    volume = mini_ccf.volume()
    for step, voxel in enumerate(mini_ccf.path_voxels(path_index)):
        volume[tuple(voxel)] = float(step + 1)
    return volume


# -- slab geometry ---------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["unnormalized", "normalized_full", "normalized_layers"]
)
def test_slab_shape_is_the_view_plus_a_depth_axis(request, mini_ccf, fixture_name):
    projector = request.getfixturevalue(fixture_name)
    slab = projector.project_volume(mini_ccf.volume())
    assert slab.shape == mini_ccf.view_size + (mini_ccf.paths.shape[1],)


def test_both_hemispheres_doubles_the_first_dimension(mini_ccf):
    projector = _projector(mini_ccf, hemisphere="both")
    slab = projector.project_volume(mini_ccf.volume())
    assert slab.shape == (
        mini_ccf.view_size[0] * 2,
        mini_ccf.view_size[1],
        mini_ccf.paths.shape[1],
    )


def test_both_with_view_space_reproduces_the_view_size(mini_ccf):
    projector = _projector(
        mini_ccf, hemisphere="both", view_space_for_other_hemisphere=True
    )
    slab = projector.project_volume(_ramp(mini_ccf, 0))
    assert slab.shape == mini_ccf.view_size + (mini_ccf.paths.shape[1],)
    assert slab.max() == 8.0


def test_a_volume_of_the_wrong_shape_raises(unnormalized):
    with pytest.raises(ValueError, match="must match lookup volume shape"):
        unnormalized.project_volume(np.zeros((2, 2, 2)))


# -- layer block sizes -----------------------------------------------------


def test_reference_layer_thicknesses_in_voxels(normalized_layers, mini_ccf):
    """Each layer's share of the padded length, rounded.

    With the real layer ratios and a padded length of 12 these come to
    [1, 3, 2, 3, 2, 1], which is the partition the slab's depth axis is built
    from. They must sum to exactly the padded length or the projection cannot
    reshape.
    """
    blocks = normalized_layers.reference_layer_thicknesses_in_voxels()

    assert list(blocks.values()) == [1, 3, 2, 3, 2, 1]
    assert sum(blocks.values()) == mini_ccf.paths.shape[1]
    assert all(v > 0 for v in blocks.values())


def test_block_sizes_follow_the_reference_thicknesses(mini_ccf):
    """Doubling every reference thickness changes nothing: only ratios matter."""
    doubled = {k: v * 2 for k, v in mini_ccf.layer_thicknesses.items()}
    projector = _projector(
        mini_ccf,
        thickness_type="normalized_layers",
        layer_thicknesses=doubled,
        streamline_layer_thickness_file=mini_ccf.layer_thickness_file,
    )
    assert list(projector.reference_layer_thicknesses_in_voxels().values()) == [
        1,
        3,
        2,
        3,
        2,
        1,
    ]


# -- the three thickness types differ --------------------------------------


def test_unnormalized_keeps_the_streamline_as_it_is(unnormalized, mini_ccf):
    """Eight voxels of data, then four of padding."""
    row, col = mini_ccf.view_pixel_for_path(0)
    profile = unnormalized.project_volume(_ramp(mini_ccf, 0))[row, col, :]

    assert np.array_equal(profile[:8], np.arange(1, 9, dtype=float))
    assert np.array_equal(profile[8:], np.zeros(4))


def test_normalized_full_stretches_the_streamline_over_the_whole_depth(
    normalized_full, mini_ccf
):
    """The padding is gone: an 8-voxel streamline fills all 12 depth slots."""
    row, col = mini_ccf.view_pixel_for_path(0)
    profile = normalized_full.project_volume(_ramp(mini_ccf, 0))[row, col, :]

    assert profile[0] == pytest.approx(1.0)
    assert np.all(profile > 0)
    assert profile.max() <= 8.0


def test_the_three_thickness_types_give_different_profiles(mini_ccf):
    row, col = mini_ccf.view_pixel_for_path(0)
    volume = _ramp(mini_ccf, 0)

    profiles = [
        _projector(mini_ccf).project_volume(volume.copy())[row, col, :],
        _projector(mini_ccf, thickness_type="normalized_full").project_volume(
            volume.copy()
        )[row, col, :],
        _projector(
            mini_ccf,
            thickness_type="normalized_layers",
            layer_thicknesses=mini_ccf.layer_thicknesses,
            streamline_layer_thickness_file=mini_ccf.layer_thickness_file,
        ).project_volume(volume.copy())[row, col, :],
    ]

    assert not np.allclose(profiles[0], profiles[1])
    assert not np.allclose(profiles[1], profiles[2])
    assert not np.allclose(profiles[0], profiles[2])


def test_thickness_type_can_be_overridden_per_call(mini_ccf):
    projector = _projector(mini_ccf, thickness_type="unnormalized")
    row, col = mini_ccf.view_pixel_for_path(0)
    volume = _ramp(mini_ccf, 0)

    default = projector.project_volume(volume.copy())[row, col, :]
    overridden = projector.project_volume(
        volume.copy(), thickness_type="normalized_full"
    )[row, col, :]

    assert not np.allclose(default, overridden)


def test_an_unknown_thickness_type_override_raises(unnormalized, mini_ccf):
    with pytest.raises(ValueError, match="Unknown thickness type"):
        unnormalized.project_volume(mini_ccf.volume(), thickness_type="normalised_full")


# -- layers deliberately absent --------------------------------------------


def test_an_absent_layer_is_left_empty_in_the_slab(normalized_layers, mini_ccf):
    """The documented behaviour: layers not present in a region are left as
    gaps, rather than the surrounding layers closing over them.

    Streamline ``absent_layer_path_index`` has no layer 4, and layer 4's block
    is the two depth slots after layers 1 and 2/3 (1 + 3 = 4), so slots 4 and 5
    must be exactly zero while every other slot carries data.
    """
    path_index = mini_ccf.absent_layer_path_index
    row, col = mini_ccf.view_pixel_for_path(path_index)

    blocks = normalized_layers.reference_layer_thicknesses_in_voxels()
    keys = list(blocks)
    absent_at = keys.index(mini_ccf.absent_layer_key)
    start = sum(list(blocks.values())[:absent_at])
    stop = start + blocks[mini_ccf.absent_layer_key]

    profile = normalized_layers.project_volume(_ramp(mini_ccf, path_index))[row, col, :]

    assert np.array_equal(profile[start:stop], np.zeros(stop - start))
    assert np.all(profile[:start] > 0)
    assert np.all(profile[stop:] > 0)


def test_a_streamline_with_every_layer_has_no_gap(normalized_layers, mini_ccf):
    path_index = 0
    assert path_index != mini_ccf.absent_layer_path_index
    row, col = mini_ccf.view_pixel_for_path(path_index)

    profile = normalized_layers.project_volume(_ramp(mini_ccf, path_index))[row, col, :]

    assert np.all(profile > 0)


# -- documented error conditions -------------------------------------------


def test_an_invalid_thickness_type_raises_at_construction(mini_ccf):
    with pytest.raises(ValueError, match="not in allowed values"):
        _projector(mini_ccf, thickness_type="normalised_layers")


def test_normalized_layers_requires_a_streamline_thickness_file(mini_ccf):
    with pytest.raises(ValueError, match="streamline_layer_thickness_file"):
        _projector(
            mini_ccf,
            thickness_type="normalized_layers",
            layer_thicknesses=mini_ccf.layer_thicknesses,
        )


def test_normalized_layers_requires_reference_thicknesses(mini_ccf):
    with pytest.raises(ValueError, match="layer_thicknesses"):
        _projector(
            mini_ccf,
            thickness_type="normalized_layers",
            streamline_layer_thickness_file=mini_ccf.layer_thickness_file,
        )


def test_per_streamline_thicknesses_are_ordered_to_match_the_view(
    normalized_layers, mini_ccf
):
    """Loaded rows are reindexed by path ordering, like ``paths`` itself."""
    for key in normalized_layers.ISOCORTEX_LAYER_KEYS:
        loaded = normalized_layers.path_layer_thickness[key]
        assert loaded.shape[0] == mini_ccf.view_lookup.shape[0]
        for row, path_index in enumerate(mini_ccf.view_path_indices):
            assert np.allclose(
                loaded[row], mini_ccf.path_layer_thickness[key][path_index]
            )
