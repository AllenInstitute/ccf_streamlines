"""``IsocortexCoordinateProjector``: depths, 2-D placement, and hemispheres.

The mini-CCF's streamlines run straight down +y, so a query at a voxel corner
has an offset residual of exactly zero and the projected view coordinate is an
exact integer -- assertions here are equalities, not tolerances.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import IsocortexCoordinateProjector

NO_PROJECTION_FILE = (
    "constructing without `projection_file` and then calling "
    "`project_coordinates` raises AttributeError deep inside "
    "`_calculate_2d_coordinates` instead of a clear error at the call; remove "
    "this marker when it is fixed"
)


def _projector(mini_ccf, **kwargs):
    kwargs.setdefault("projection_file", mini_ccf.view_lookup_file)
    kwargs.setdefault("layer_thicknesses", mini_ccf.layer_thicknesses)
    kwargs.setdefault("streamline_layer_thickness_file", mini_ccf.layer_thickness_file)
    return IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        resolution=mini_ccf.resolution,
        **kwargs,
    )


@pytest.fixture
def projector(mini_ccf):
    return _projector(mini_ccf)


#: Queries are made with drop-outside on, which sidesteps the empty-search
#: crash pinned in tests/test_pinned_contributions.py. Once #13 lands this can
#: be dropped.
DROP = dict(drop_voxels_outside_view_streamlines=True)


# -- depths ----------------------------------------------------------------


@pytest.mark.parametrize("step", [0, 1, 3, 7])
def test_unnormalized_depth_is_the_distance_along_the_streamline(
    projector, mini_ccf, step
):
    """Voxel centres sit half a voxel below the corner the query lands on, so a
    query at step *n* is *n* - 0.5 voxels along, floored at the pia end."""
    coord = mini_ccf.coord_on_path(0, step).reshape(1, 3)

    depth = projector.project_depths(coord, thickness_type="unnormalized")

    assert depth[0] == pytest.approx(max(step - 0.5, 0.0))


def test_depth_in_microns_is_the_voxel_depth_scaled(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 4).reshape(1, 3)

    in_voxels = projector.project_depths(coord, scale="voxels")
    in_microns = projector.project_depths(coord, scale="microns")

    assert in_microns[0] == pytest.approx(in_voxels[0] * mini_ccf.resolution[1])


def test_normalized_full_rescales_to_the_padded_length(projector, mini_ccf):
    """Depth becomes a fraction of the streamline, times the padded length.

    The streamline spans 7 voxel centres, so a query 2.5 along is 2.5/7 of the
    way down, which over a padded length of 12 is 30/7.
    """
    coord = mini_ccf.coord_on_path(0, 3).reshape(1, 3)

    depth = projector.project_depths(coord, thickness_type="normalized_full")

    padded = mini_ccf.paths.shape[1]
    assert depth[0] == pytest.approx(2.5 / (mini_ccf.path_length - 1) * padded)


def test_normalized_layers_spans_the_padded_length(projector, mini_ccf):
    """Pia is 0 and the deepest queryable point approaches the padded length."""
    padded = mini_ccf.paths.shape[1]

    at_pia = projector.project_depths(
        mini_ccf.coord_on_path(0, 0).reshape(1, 3), thickness_type="normalized_layers"
    )
    deep = projector.project_depths(
        mini_ccf.coord_on_path(0, 7).reshape(1, 3), thickness_type="normalized_layers"
    )

    assert at_pia[0] == pytest.approx(0.0, abs=1e-9)
    assert 0 < deep[0] <= padded


def test_normalized_layers_is_monotonic_down_the_streamline(projector, mini_ccf):
    coords = np.vstack([mini_ccf.coord_on_path(0, s) for s in range(8)])

    depths = projector.project_depths(coords, thickness_type="normalized_layers")

    assert np.all(np.diff(depths) > 0)


def test_the_three_thickness_types_give_different_depths(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 4).reshape(1, 3)

    depths = {
        t: projector.project_depths(coord, thickness_type=t)[0]
        for t in ("unnormalized", "normalized_full", "normalized_layers")
    }

    assert len(set(depths.values())) == 3


def test_only_the_ratios_of_reference_thicknesses_affect_the_result(mini_ccf):
    """Scaling every reference layer thickness by the same factor changes
    nothing, because the result is rescaled to the padded streamline length.

    This is the weaker, true form of the concern raised in issue #10: the
    absolute total is discarded, not double-normalized.
    """
    coord = mini_ccf.coord_on_path(0, 4).reshape(1, 3)
    doubled = {k: v * 2 for k, v in mini_ccf.layer_thicknesses.items()}

    baseline = _projector(mini_ccf).project_depths(
        coord, thickness_type="normalized_layers"
    )
    scaled = _projector(mini_ccf, layer_thicknesses=doubled).project_depths(
        coord, thickness_type="normalized_layers"
    )

    assert baseline[0] == pytest.approx(scaled[0])


def test_changing_the_ratios_does_change_the_result(mini_ccf):
    """The complement of the test above: ratios are not ignored."""
    coord = mini_ccf.coord_on_path(0, 4).reshape(1, 3)
    skewed = dict(mini_ccf.layer_thicknesses)
    skewed["Isocortex layer 1"] = skewed["Isocortex layer 1"] * 10

    baseline = _projector(mini_ccf).project_depths(
        coord, thickness_type="normalized_layers"
    )
    changed = _projector(mini_ccf, layer_thicknesses=skewed).project_depths(
        coord, thickness_type="normalized_layers"
    )

    assert baseline[0] != pytest.approx(changed[0])


def test_a_coordinate_outside_cortex_has_no_depth(projector, mini_ccf):
    """The dorso-ventral planes at each end carry no streamline voxels."""
    outside = (np.array([3, 0, 1]) * np.array(mini_ccf.resolution)).astype(float)

    depth = projector.project_depths(outside.reshape(1, 3))

    assert np.isnan(depth[0])


def test_an_invalid_scale_raises(projector, mini_ccf):
    with pytest.raises(ValueError, match="`scale` must be"):
        projector.project_depths(mini_ccf.coord_on_path(0, 1).reshape(1, 3), scale="mm")


# -- 2-D placement ---------------------------------------------------------


def test_a_coordinate_on_a_streamline_lands_exactly_on_its_view_pixel(
    projector, mini_ccf
):
    """No tolerance: the geometric offset residual at a voxel corner is zero."""
    for path_index in mini_ccf.in_view_path_indices:
        coord = mini_ccf.coord_on_path(int(path_index), 3).reshape(1, 3)

        result = projector.project_coordinates(coord, hemisphere="left", **DROP)

        expected_row, expected_col = mini_ccf.view_pixel_for_path(int(path_index))
        assert (result[0, 0], result[0, 1]) == (expected_row, expected_col)


def test_projected_coordinates_are_x_y_and_depth(projector, mini_ccf):
    coords = np.vstack([mini_ccf.coord_on_path(0, s) for s in (1, 3, 5)])

    result = projector.project_coordinates(coords, hemisphere="left", **DROP)

    assert result.shape == (3, 3)
    depths = projector.project_depths(coords)
    assert np.array_equal(result[:, 2], depths)


def test_microns_scale_multiplies_by_the_resolution(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 3).reshape(1, 3)

    in_voxels = projector.project_coordinates(
        coord, scale="voxels", hemisphere="left", **DROP
    )
    in_microns = projector.project_coordinates(
        coord, scale="microns", hemisphere="left", **DROP
    )

    assert in_microns[0, 0] == pytest.approx(in_voxels[0, 0] * mini_ccf.resolution[2])
    assert in_microns[0, 1] == pytest.approx(in_voxels[0, 1] * mini_ccf.resolution[0])


def test_a_coordinate_outside_cortex_projects_to_nan(projector, mini_ccf):
    """Unmappable points come back as a documented sentinel, not a wrong number."""
    outside = (np.array([3, 0, 1]) * np.array(mini_ccf.resolution)).astype(float)

    result = projector.project_coordinates(outside.reshape(1, 3), hemisphere="left", **DROP)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])


# -- drop_voxels_outside_view_streamlines ----------------------------------


def test_dropping_outside_view_coordinates_gives_nan(projector, mini_ccf):
    out_of_view = mini_ccf.coord_on_path(
        int(mini_ccf.out_of_view_path_indices[0]), 3
    ).reshape(1, 3)

    result = projector.project_coordinates(
        out_of_view, hemisphere="left", drop_voxels_outside_view_streamlines=True
    )

    assert np.isnan(result[0, 0])
    # ...but the depth is still known, because the streamline exists.
    assert not np.isnan(result[0, 2])


def test_keeping_outside_view_coordinates_snaps_to_the_nearest_streamline(
    projector, mini_ccf
):
    """With the default, a coordinate off the view is placed at the nearest
    streamline the view does use, rather than discarded."""
    in_view = mini_ccf.coord_on_path(int(mini_ccf.in_view_path_indices[0]), 3)
    out_of_view = mini_ccf.coord_on_path(int(mini_ccf.out_of_view_path_indices[0]), 3)
    coords = np.vstack([in_view, out_of_view])

    snapped = projector.project_coordinates(
        coords, hemisphere="left", drop_voxels_outside_view_streamlines=False
    )

    assert not np.isnan(snapped[1, 0])
    # The snapped point lands on a pixel the view actually contains.
    pixel = (snapped[1, 0], snapped[1, 1])
    all_pixels = {
        tuple(np.unravel_index(int(f), mini_ccf.view_size))
        for f in mini_ccf.view_lookup[:, 0]
    }
    assert pixel in all_pixels


# -- hemispheres -----------------------------------------------------------


def test_left_keeps_everything_on_the_left(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 3).reshape(1, 3)
    expected_row, _ = mini_ccf.view_pixel_for_path(0)

    result = projector.project_coordinates(coord, hemisphere="left", **DROP)

    assert result[0, 0] == expected_row


def test_right_reflects_everything_to_the_right(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 3).reshape(1, 3)
    expected_row, _ = mini_ccf.view_pixel_for_path(0)

    result = projector.project_coordinates(coord, hemisphere="right", **DROP)

    assert result[0, 0] == mini_ccf.view_size[0] - expected_row


def test_both_keeps_each_coordinate_on_its_own_side(projector, mini_ccf):
    """A right-hemisphere coordinate is placed in the right half of the view.

    Note the coordinate projector reflects with ``z_size - z``; the volume
    projectors use ``np.flip``, which is ``z_size - 1 - z``. See
    ``test_projector_2d.test_the_two_hemisphere_mirroring_conventions_differ_by_one_voxel``.
    """
    z_size = mini_ccf.volume_shape[2]
    voxel = mini_ccf.path_voxels(0)[3].copy()
    left_coord = (voxel * np.array(mini_ccf.resolution)).astype(float)
    voxel[2] = z_size - voxel[2]
    right_coord = (voxel * np.array(mini_ccf.resolution)).astype(float)

    result = projector.project_coordinates(
        np.vstack([left_coord, right_coord]), hemisphere="both", **DROP
    )

    expected_row, _ = mini_ccf.view_pixel_for_path(0)
    assert result[0, 0] == expected_row
    assert result[1, 0] == 2 * mini_ccf.view_size[0] - expected_row


def test_both_mirrored_swaps_the_sides(projector, mini_ccf):
    z_size = mini_ccf.volume_shape[2]
    voxel = mini_ccf.path_voxels(0)[3].copy()
    left_coord = (voxel * np.array(mini_ccf.resolution)).astype(float)
    voxel[2] = z_size - voxel[2]
    right_coord = (voxel * np.array(mini_ccf.resolution)).astype(float)
    coords = np.vstack([left_coord, right_coord])

    plain = projector.project_coordinates(coords, hemisphere="both", **DROP)
    mirrored = projector.project_coordinates(coords, hemisphere="both_mirrored", **DROP)

    max_x = mini_ccf.view_size[0]
    assert mirrored[0, 0] == 2 * max_x - plain[0, 0]
    assert mirrored[1, 0] == 2 * max_x - plain[1, 0]


def test_view_space_shifts_where_the_right_hemisphere_lands(projector, mini_ccf):
    coord = mini_ccf.coord_on_path(0, 3).reshape(1, 3)

    without = projector.project_coordinates(coord, hemisphere="right", **DROP)
    with_space = projector.project_coordinates(
        coord, hemisphere="right", view_space_for_other_hemisphere=2, **DROP
    )

    assert with_space[0, 0] == without[0, 0] - 2


def test_an_invalid_hemisphere_raises(projector, mini_ccf):
    with pytest.raises(ValueError, match="`hemisphere` must be"):
        projector.project_coordinates(
            mini_ccf.coord_on_path(0, 1).reshape(1, 3), hemisphere="middle"
        )


def test_an_invalid_scale_raises_for_project_coordinates(projector, mini_ccf):
    with pytest.raises(ValueError, match="`scale` must be"):
        projector.project_coordinates(
            mini_ccf.coord_on_path(0, 1).reshape(1, 3), scale="mm"
        )


def test_an_unknown_named_view_raises(projector, mini_ccf):
    with pytest.raises(ValueError, match="unknown string option"):
        projector.project_coordinates(
            mini_ccf.coord_on_path(0, 1).reshape(1, 3),
            view_space_for_other_hemisphere="flatmap_buttrfly",
        )


# -- construction ----------------------------------------------------------


def test_depths_work_without_a_projection_file(mini_ccf):
    """Documented: the class can be used for depth alone."""
    projector = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        resolution=mini_ccf.resolution,
    )

    depth = projector.project_depths(mini_ccf.coord_on_path(0, 3).reshape(1, 3))

    assert depth[0] == pytest.approx(2.5)


def test_layer_thicknesses_are_optional_when_not_used(mini_ccf):
    projector = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        resolution=mini_ccf.resolution,
    )
    assert projector.path_layer_thickness is None


@pytest.mark.xfail(strict=True, reason=NO_PROJECTION_FILE)
def test_projecting_without_a_projection_file_raises_a_clear_error(mini_ccf):
    """Currently an AttributeError about a missing `view_lookup` attribute,
    raised from inside a private method -- not something a caller can act on."""
    projector = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        resolution=mini_ccf.resolution,
    )

    with pytest.raises(ValueError, match="projection_file"):
        projector.project_coordinates(mini_ccf.coord_on_path(0, 3).reshape(1, 3))
