"""``Isocortex2dProjector``: construction, aggregation, and hemispheres.

Built from mini-CCF files through the public constructor, so this exercises
the loading code as well as the projection -- and the loading code is where a
substantial share of the known defects live.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import HEMISPHERE_SPACE_VIEW_LOOKUP, Isocortex2dProjector

MUTATES_INPUT = (
    "`project_volume` writes a sentinel into volume.flat[0] for max/min and "
    "never restores it, silently corrupting the caller's array; remove this "
    "marker when it is fixed"
)
UNKNOWN_KIND_SILENT = (
    "`_project_volume_to_view` has no else branch, so an unrecognised `kind` "
    "returns an all-zeros view instead of raising; remove this marker when it "
    "is fixed"
)
MISSING_DATASET = (
    "`project_path_ordered_data` reads the 3-D 'volume lookup' dataset, which "
    "does not exist in the current-generation surface paths file; remove this "
    "marker when it is fixed"
)


@pytest.fixture
def projector(mini_ccf):
    return Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="left"
    )


@pytest.fixture
def ramp_volume(mini_ccf):
    """A volume holding 1..8 down streamline 0 and nothing else."""
    volume = mini_ccf.volume()
    for step, voxel in enumerate(mini_ccf.path_voxels(0)):
        volume[tuple(voxel)] = float(step + 1)
    return volume


# -- construction ----------------------------------------------------------


def test_attributes_are_decoded_from_the_projection_file(projector, mini_ccf):
    assert tuple(projector.view_size) == mini_ccf.view_size
    # `spacing` is stored as byte strings and must come back as ints.
    assert projector.resolution == mini_ccf.resolution
    assert all(isinstance(r, int) for r in projector.resolution)


def test_volume_shape_comes_from_the_flat_lookup_attribute(projector, mini_ccf):
    assert tuple(projector.volume_shape) == mini_ccf.volume_shape


def test_paths_are_ordered_to_match_the_view(projector, mini_ccf):
    """One row of `paths` per view-lookup row, resolved through the sentinel
    flat lookup."""
    assert projector.paths.shape[0] == mini_ccf.view_lookup.shape[0]
    assert np.array_equal(projector.path_ordering, mini_ccf.view_path_indices)
    for row, path_index in enumerate(mini_ccf.view_path_indices):
        assert np.array_equal(projector.paths[row], mini_ccf.paths[path_index])


def test_invalid_hemisphere_raises(mini_ccf):
    with pytest.raises(ValueError, match="is not allowed"):
        Isocortex2dProjector(
            mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="middle"
        )


# -- view_space_for_other_hemisphere, in each documented form ---------------


def test_view_space_false_is_zero(mini_ccf):
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
        view_space_for_other_hemisphere=False,
    )
    assert proj.view_space_for_other_hemisphere == 0


def test_view_space_true_is_half_the_view(mini_ccf):
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
        view_space_for_other_hemisphere=True,
    )
    assert proj.view_space_for_other_hemisphere == mini_ccf.view_size[0] // 2


@pytest.mark.parametrize("name,expected", sorted(HEMISPHERE_SPACE_VIEW_LOOKUP.items()))
def test_view_space_named_view_resolves_to_its_preset(mini_ccf, name, expected):
    """Presets are absolute counts sized for full-resolution views -- cropping a
    miniature view by 110 or 390 would empty it -- so this asserts the resolved
    value rather than attempting a round trip."""
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
        view_space_for_other_hemisphere=name,
    )
    assert proj.view_space_for_other_hemisphere == expected


def test_view_space_integer_is_used_as_is(mini_ccf):
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
        view_space_for_other_hemisphere=3,
    )
    assert proj.view_space_for_other_hemisphere == 3


def test_unknown_named_view_raises_at_construction(mini_ccf):
    """A misspelled view name must fail here, not produce a wrong picture."""
    with pytest.raises(ValueError, match="unknown string option"):
        Isocortex2dProjector(
            mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
            view_space_for_other_hemisphere="flatmap_buttrfly",
        )


# -- aggregation kinds -----------------------------------------------------


def test_max_projection_takes_the_brightest_voxel_on_the_streamline(
    projector, mini_ccf, ramp_volume
):
    row, col = mini_ccf.view_pixel_for_path(0)
    result = projector.project_volume(ramp_volume, kind="max")

    assert result.shape == mini_ccf.view_size
    assert result[row, col] == 8.0


def test_min_projection_takes_the_dimmest_voxel_on_the_streamline(
    projector, mini_ccf, ramp_volume
):
    row, col = mini_ccf.view_pixel_for_path(0)
    assert projector.project_volume(ramp_volume, kind="min")[row, col] == 1.0


@pytest.mark.parametrize("kind", ["mean", "average"])
def test_mean_projection_averages_over_the_streamline(
    projector, mini_ccf, ramp_volume, kind
):
    """1..8 averages to 4.5. Padding is excluded, not counted as zero."""
    row, col = mini_ccf.view_pixel_for_path(0)
    assert projector.project_volume(ramp_volume, kind=kind)[row, col] == pytest.approx(4.5)


def test_sum_projection_totals_the_streamline(projector, mini_ccf, ramp_volume):
    row, col = mini_ccf.view_pixel_for_path(0)
    assert projector.project_volume(ramp_volume, kind="sum")[row, col] == pytest.approx(36.0)


def test_the_four_aggregations_differ(projector, mini_ccf, ramp_volume):
    row, col = mini_ccf.view_pixel_for_path(0)
    values = {
        kind: projector.project_volume(ramp_volume.copy(), kind=kind)[row, col]
        for kind in ("max", "min", "mean", "sum")
    }
    assert len(set(values.values())) == 4


def test_pixels_with_no_streamline_stay_zero(projector, mini_ccf, ramp_volume):
    result = projector.project_volume(ramp_volume, kind="max")
    covered = {mini_ccf.view_pixel_for_path(0), *mini_ccf.view_pixels_for_path(0)}
    for r in range(mini_ccf.view_size[0]):
        for c in range(mini_ccf.view_size[1]):
            if (r, c) not in covered:
                assert result[r, c] == 0.0


# -- hemispheres -----------------------------------------------------------


def test_left_hemisphere_output_is_the_view_size(mini_ccf, ramp_volume):
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="left"
    )
    assert proj.project_volume(ramp_volume).shape == mini_ccf.view_size


def test_both_hemispheres_doubles_the_first_dimension(mini_ccf, ramp_volume):
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="both"
    )
    result = proj.project_volume(ramp_volume)
    assert result.shape == (mini_ccf.view_size[0] * 2, mini_ccf.view_size[1])


def test_both_with_view_space_reproduces_the_original_view_size(mini_ccf, ramp_volume):
    """Cropping each half by half the view and concatenating gets back to the
    view's own size -- which is the point of the option."""
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file,
        hemisphere="both", view_space_for_other_hemisphere=True,
    )
    result = proj.project_volume(ramp_volume)
    assert result.shape == mini_ccf.view_size
    # All content is in the retained half, so nothing was lost.
    assert result.max() == 8.0


def test_right_hemisphere_flips_the_first_dimension(mini_ccf):
    """A voxel on the right maps to the mirrored row of the view."""
    volume = mini_ccf.volume()
    x, y, z = mini_ccf.path_voxels(0)[3]
    z_size = mini_ccf.volume_shape[2]
    # `project_volume` mirrors with np.flip, which maps index z to z_size-1-z.
    volume[x, y, z_size - 1 - z] = 5.0

    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="right"
    )
    result = proj.project_volume(volume)

    lit = {tuple(p) for p in np.argwhere(result > 0)}
    expected = {
        (mini_ccf.view_size[0] - 1 - r, c)
        for r, c in mini_ccf.view_pixels_for_path(0)
    }
    assert lit == expected


def test_the_two_hemisphere_mirroring_conventions_differ_by_one_voxel(mini_ccf):
    """Pinned characterization, not an assertion that either is correct.

    ``Isocortex2dProjector`` mirrors a volume with ``np.flip(volume, axis=2)``,
    which maps voxel z to ``z_size - 1 - z``. ``IsocortexCoordinateProjector``
    mirrors coordinates with ``z_size - z``. The two therefore disagree by one
    voxel about which side of the midline a given voxel is on.

    Which is intended is not knowable from the code or the documentation, so
    this records the current behaviour rather than asserting a correct answer.
    Overlaying projected coordinates on a projected volume is off by one voxel
    laterally until it is resolved.
    """
    z_size = mini_ccf.volume_shape[2]
    x, y, z = mini_ccf.path_voxels(0)[3]

    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="right"
    )

    volume_convention = mini_ccf.volume()
    volume_convention[x, y, z_size - 1 - z] = 5.0
    assert proj.project_volume(volume_convention).max() == 5.0

    coordinate_convention = mini_ccf.volume()
    coordinate_convention[x, y, z_size - z] = 5.0
    assert proj.project_volume(coordinate_convention).max() == 0.0


# -- documented error conditions -------------------------------------------


def test_a_volume_of_the_wrong_shape_raises(projector):
    with pytest.raises(ValueError, match="must match lookup volume shape"):
        projector.project_volume(np.zeros((2, 2, 2)))


def test_a_volume_of_a_non_numeric_dtype_raises(projector, mini_ccf):
    with pytest.raises(ValueError, match="integer or float"):
        projector.project_volume(mini_ccf.volume(dtype=bool))


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32, np.float64])
def test_integer_and_float_volumes_are_both_accepted(projector, mini_ccf, dtype):
    volume = mini_ccf.volume(dtype=dtype)
    volume[tuple(mini_ccf.path_voxels(0)[2])] = 3
    row, col = mini_ccf.view_pixel_for_path(0)

    result = projector.project_volume(volume, kind="max")

    assert result.dtype == dtype
    assert result[row, col] == 3


# -- defects ---------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=UNKNOWN_KIND_SILENT)
def test_an_unrecognised_kind_raises(projector, mini_ccf, ramp_volume):
    """A typo currently returns an array of zeros, which looks like real data."""
    with pytest.raises(ValueError):
        projector.project_volume(ramp_volume, kind="maximum")


@pytest.mark.xfail(strict=True, reason=MUTATES_INPUT)
@pytest.mark.parametrize("kind", ["max", "min"])
def test_projection_does_not_modify_the_callers_volume(projector, mini_ccf, kind):
    """Callers must not have to defensively copy before projecting."""
    volume = mini_ccf.volume()
    volume[tuple(mini_ccf.path_voxels(0)[2])] = 3.0
    before = volume.copy()

    projector.project_volume(volume, kind=kind)

    assert np.array_equal(volume, before)


@pytest.mark.xfail(strict=True, reason=MUTATES_INPUT)
def test_projecting_both_hemispheres_does_not_modify_the_callers_volume(mini_ccf):
    """The second pass writes through an ``np.flip`` view, so the far corner is
    clobbered as well as the first voxel."""
    proj = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="both"
    )
    volume = mini_ccf.volume()
    before = volume.copy()

    proj.project_volume(volume, kind="max")

    assert np.array_equal(volume, before)


@pytest.mark.xfail(strict=True, reason=MISSING_DATASET)
def test_project_path_ordered_data(projector, mini_ccf):
    """A documented public method that cannot run against current assets.

    It reads the 3-D ``volume lookup`` dataset, which the v3 surface-paths file
    does not contain -- only the flattened form. The projector already holds
    ``path_ordering``, which is the same information.
    """
    data = np.arange(mini_ccf.paths.shape[0], dtype=float)

    result = projector.project_path_ordered_data(data)

    row, col = mini_ccf.view_pixel_for_path(0)
    assert result.shape == mini_ccf.view_size
    assert result[row, col] == 0.0
