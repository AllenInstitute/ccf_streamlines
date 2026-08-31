"""``IsocortexCoordinateProjector``: depths, 2-D placement, and hemispheres.

The mini-CCF's streamlines run straight down +y, so a query at a voxel corner
has an offset residual of exactly zero and the projected view coordinate is an
exact integer -- assertions here are equalities, not tolerances.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import IsocortexCoordinateProjector


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


def _unambiguous_paths(mini_ccf):
    """Paths the view references exactly once.

    A streamline that two view pixels both show has no single correct answer
    here until #11 lands: which of the tied pixels comes back depends on how
    the platform's sort orders equal keys, and it genuinely differs between
    architectures -- GitHub's x86-64 runners and its arm64 runners disagree on
    this fixture. Asserting one of them would only re-test the pinned
    contribution, badly.

    So the exactness tests use streamlines with an unambiguous pixel, and the
    tied case stays in ``tests/test_pinned_contributions.py`` where it belongs.
    """
    return [
        int(p)
        for p in mini_ccf.in_view_path_indices
        if len(mini_ccf.view_pixels_for_path(int(p))) == 1
    ]


@pytest.fixture
def unambiguous_path(mini_ccf):
    paths = _unambiguous_paths(mini_ccf)
    assert paths, "the fixture must have at least one untied in-view streamline"
    return paths[0]


def _mirrored_voxel(mini_ccf, voxel):
    """The voxel on the other side of the midline, mirrored correctly.

    Voxel z spans ``[z, z+1)`` with centre ``z + 0.5``; reflecting that centre
    about the midline plane at ``z_size / 2`` gives ``(z_size - 1 - z) + 0.5``,
    the centre of voxel ``z_size - 1 - z``. That is what ``np.flip`` does, and
    it is what the coordinate projector's *micron* reflection does.

    It is also what the coordinate projector's *voxel* reflection does, since
    #27; it used to be ``z_size - z``, one voxel out. Tests here always built
    their input with the correct convention, so none of them baked the defect
    into its own setup.
    """
    mirrored = np.array(voxel).copy()
    mirrored[2] = mini_ccf.volume_shape[2] - 1 - mirrored[2]
    return mirrored


#: Queries are made with drop-outside on, which sidesteps the empty-search
#: crash pinned in tests/test_pinned_contributions.py. Once #13 lands this can
#: be dropped.
DROP = {"drop_voxels_outside_view_streamlines": True}


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
    for path_index in _unambiguous_paths(mini_ccf):
        coord = mini_ccf.coord_on_path(path_index, 3).reshape(1, 3)

        result = projector.project_coordinates(coord, hemisphere="left", **DROP)

        expected_row, expected_col = mini_ccf.view_pixel_for_path(path_index)
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

    result = projector.project_coordinates(
        outside.reshape(1, 3), hemisphere="left", **DROP
    )

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


def test_left_keeps_everything_on_the_left(projector, mini_ccf, unambiguous_path):
    coord = mini_ccf.coord_on_path(unambiguous_path, 3).reshape(1, 3)
    expected_row, _ = mini_ccf.view_pixel_for_path(unambiguous_path)

    result = projector.project_coordinates(coord, hemisphere="left", **DROP)

    assert result[0, 0] == expected_row


def test_right_reflects_everything_to_the_right(projector, mini_ccf, unambiguous_path):
    coord = mini_ccf.coord_on_path(unambiguous_path, 3).reshape(1, 3)
    expected_row, _ = mini_ccf.view_pixel_for_path(unambiguous_path)

    result = projector.project_coordinates(coord, hemisphere="right", **DROP)

    assert result[0, 0] == mini_ccf.view_size[0] - expected_row


def test_both_keeps_each_coordinate_on_its_own_side(
    projector, mini_ccf, unambiguous_path
):
    """A right-hemisphere coordinate is placed in the right half of the view.

    The input is built with the correct mirror, ``z_size - 1 - z``, so this
    test does not depend on the reflection defect being present.

    The left coordinate is asserted exactly. The right one is only asserted to
    land in the right half, because its exact position is currently one voxel
    short of the mirrored row: the voxel reflection resolves it to the
    neighbouring streamline. Once that is fixed it will sit exactly at
    ``2 * max_x - expected_row``. The defect is pinned under "reflection
    consistency" at the bottom of this file, and is not re-asserted here.
    """
    voxel = mini_ccf.path_voxels(unambiguous_path)[3]
    resolution = np.array(mini_ccf.resolution)
    left_coord = (voxel * resolution).astype(float)
    right_coord = (_mirrored_voxel(mini_ccf, voxel) * resolution).astype(float)

    result = projector.project_coordinates(
        np.vstack([left_coord, right_coord]), hemisphere="both", **DROP
    )

    expected_row, _ = mini_ccf.view_pixel_for_path(unambiguous_path)
    max_x = mini_ccf.view_size[0]

    assert result[0, 0] == expected_row
    assert result[0, 0] < max_x, "the left coordinate stays in the left half"
    assert result[1, 0] > max_x, "the right coordinate moves to the right half"


def test_both_mirrored_swaps_the_sides(projector, mini_ccf, unambiguous_path):
    """The relation asserted here holds whichever mirror the input uses, but
    the input is built with the correct one anyway."""
    voxel = mini_ccf.path_voxels(unambiguous_path)[3]
    resolution = np.array(mini_ccf.resolution)
    left_coord = (voxel * resolution).astype(float)
    right_coord = (_mirrored_voxel(mini_ccf, voxel) * resolution).astype(float)
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


# -- reflection consistency -------------------------------------------------
#
# Right-hemisphere data is mirrored onto the left before anything is looked up,
# and `_get_collapsed_voxels_and_surface_voxels` does it twice -- once for the
# voxel, which selects the streamline, and once for the micron coordinate,
# which measures depth along it. The two must land on the same voxel, or the
# depth is measured along a neighbouring streamline.
#
#     voxels[..., 2]         = z_size - 1 - z
#     reflect_coords[..., 2] = z_size * resolution - u
#
# Voxel z spans [z, z+1) with centre z + 0.5, and mirroring that centre about
# the midline plane at z_size/2 gives (z_size - 1 - z) + 0.5, the centre of
# voxel z_size - 1 - z. The volume projectors agree -- `np.flip(volume,
# axis=2)` is exactly z_size-1-z.
#
# The voxel line used to read `z_size - z`, one voxel out. The two forms agree
# only when a coordinate sits exactly on a voxel boundary, which is precisely
# what `coord_on_path` returns -- deliberate elsewhere, since a corner makes
# the geometric offset exactly zero and view pixels come out as exact integers,
# but it meant the rest of this file could not see the mismatch. The tests
# below query voxel centres instead.


def _voxel_centre(mini_ccf, voxel):
    """Micron coordinate at the centre of a voxel, not at its corner."""
    return (np.array(voxel) + 0.5) * np.array(mini_ccf.resolution)


def test_the_voxel_reflection_is_where_the_reflected_coordinate_lands(
    projector, mini_ccf
):
    """Inside a voxel the reflection is ``z_size - 1 - z``, the np.flip form.

    The projector does not use that formula -- it takes the voxel the mirrored
    micron coordinate falls in, so the two cannot disagree -- but for any point
    inside a voxel the two are the same thing. The superseded ``z_size - z`` is
    one voxel out everywhere here, which is what made depth get measured along
    a neighbouring streamline.
    """
    from ccf_streamlines.coordinates import coordinates_to_voxels

    z_size = mini_ccf.volume_shape[2]
    res_z = mini_ccf.resolution[2]

    for z in range(z_size // 2 + 1, z_size):
        for frac in (0.01, 0.5, 0.99):
            inside = (z + frac) * res_z
            by_coordinate = coordinates_to_voxels(
                np.array([[0.0, 0.0, z_size * res_z - inside]]), mini_ccf.resolution
            )[0, 2]

            assert z_size - 1 - z == by_coordinate
            assert z_size - z != by_coordinate  # the superseded convention


@pytest.mark.parametrize("frac", [0.0, 0.01, 0.25, 0.5, 0.75, 0.99])
def test_mirror_image_points_have_the_same_depth(projector, mini_ccf, frac):
    """The invariant the whole reflection exists to preserve.

    ``u`` and ``z_size * resolution - u`` are the same point reflected about
    the midline, so they lie at the same depth in cortex. Checked at several
    positions within the voxel, including ``frac=0`` -- exactly on a boundary,
    which is what a caller passing voxel index times resolution produces, and
    the one case a voxel-space reflection formula gets wrong.
    """
    z_size = mini_ccf.volume_shape[2]
    res_z = mini_ccf.resolution[2]

    x, y, _ = mini_ccf.path_voxels(0)[3]
    lateral = sorted({int(v) for v in mini_ccf.path_starts[:, 2]})

    left = np.array(
        [
            [
                (x + frac) * mini_ccf.resolution[0],
                (y + frac) * mini_ccf.resolution[1],
                (z + frac) * res_z,
            ]
            for z in lateral
        ]
    )
    right = left.copy()
    right[:, 2] = z_size * res_z - left[:, 2]

    left_depths = projector.project_depths(left)
    right_depths = projector.project_depths(right)

    assert np.array_equal(np.isnan(left_depths), np.isnan(right_depths))
    both = ~np.isnan(left_depths)
    assert both.any(), "the left-hand controls must be on streamlines"
    assert left_depths[both] == pytest.approx(right_depths[both])


def test_a_right_hemisphere_coordinate_inside_a_voxel_finds_its_streamline(
    projector, mini_ccf
):
    """A point whose mirror lies on a streamline must have a depth.

    The deepest-lateral streamline sits at ``z_s``; a right-hemisphere point at
    the centre of voxel ``z_size - 1 - z_s`` mirrors onto it exactly. The
    superseded voxel reflection sent that point to ``z_s + 1`` instead, where
    no streamline exists, so the query came back as the not-in-cortex sentinel
    for a point that is squarely on a streamline.
    """
    z_size = mini_ccf.volume_shape[2]
    lateral = sorted({int(v) for v in mini_ccf.path_starts[:, 2]})
    z_s = max(lateral)
    assert z_s + 1 not in lateral, "the off-by-one must land where no streamline is"

    x, y, _ = mini_ccf.path_voxels(0)[3]
    left = _voxel_centre(mini_ccf, (x, y, z_s)).reshape(1, 3)
    right = _voxel_centre(mini_ccf, (x, y, z_size - 1 - z_s)).reshape(1, 3)

    left_depth = projector.project_depths(left)[0]
    right_depth = projector.project_depths(right)[0]

    assert not np.isnan(left_depth), "the left-hand control must be on a streamline"
    assert not np.isnan(right_depth)
    assert right_depth == pytest.approx(left_depth)


def test_a_right_hemisphere_coordinate_outside_cortex_stays_outside(
    projector, mini_ccf
):
    """The complement: a point whose mirror has no streamline must have none.

    The outermost lateral voxel mirrors onto ``z = 0``, where the fixture has
    no streamline. The superseded voxel reflection sent it to ``z = 1``
    instead, so a depth came back for a point that is not on any streamline.
    """
    z_size = mini_ccf.volume_shape[2]
    lateral = sorted({int(v) for v in mini_ccf.path_starts[:, 2]})
    assert 0 not in lateral

    x, y, _ = mini_ccf.path_voxels(0)[3]
    outside = _voxel_centre(mini_ccf, (x, y, z_size - 1)).reshape(1, 3)

    assert np.isnan(projector.project_depths(outside)[0])
