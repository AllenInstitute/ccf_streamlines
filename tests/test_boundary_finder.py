"""``BoundaryFinder`` against the mini-CCF's 2-D atlas.

The atlas holds two rectangles, each inset from every edge, so every contour
is a closed loop whose corners are computable by hand: a region occupying rows
r0..r1 and columns c0..c1 has its boundary at r0-0.5 .. r1+0.5 by
c0-0.5 .. c1+0.5.

``tests/test_projection.py`` covers the same class over a separate fixture and
is kept as the regression test for the empty-mask and separator fixes.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import BoundaryFinder

ABSENT_REGION_REFLECTION = (
    "reflecting a region that is in the labels file but absent from the atlas "
    "indexes the empty array it was given as if it were 2-D, raising "
    "IndexError; remove this marker when it is fixed"
)


@pytest.fixture
def finder(mini_ccf):
    return BoundaryFinder(mini_ccf.atlas_file, mini_ccf.labels_file)


def _bounds(mini_ccf, acronym):
    """The exact contour corners the region's rectangle implies."""
    rows, cols = mini_ccf.atlas_regions[acronym]
    r0, r1, _ = rows.indices(mini_ccf.atlas.shape[0])
    c0, c1, _ = cols.indices(mini_ccf.atlas.shape[1])
    return (r0 - 0.5, r1 - 1 + 0.5), (c0 - 0.5, c1 - 1 + 0.5)


# -- labels ----------------------------------------------------------------


def test_labels_are_parsed_from_ragged_whitespace(finder, mini_ccf):
    assert finder.labels_df["acronym"].tolist() == list(mini_ccf.label_acronyms)


def test_label_columns_are_named(finder):
    assert finder.labels_df.columns.tolist() == [
        "r", "g", "b", "x0", "x1", "x2", "acronym"
    ]


def test_the_atlas_is_loaded_with_its_metadata(finder, mini_ccf):
    assert finder.proj_atlas.shape == mini_ccf.atlas.shape
    assert np.array_equal(finder.proj_atlas, mini_ccf.atlas)
    assert "sizes" in finder.proj_atlas_meta


# -- region_boundaries -----------------------------------------------------


def test_boundaries_are_found_for_every_region_present_in_the_atlas(
    finder, mini_ccf
):
    boundaries = finder.region_boundaries()
    # CCC has no pixels, so it is not among the regions inferred from the atlas.
    assert set(boundaries) == set(mini_ccf.atlas_regions)


@pytest.mark.parametrize("acronym", ["AAA", "BBB"])
def test_a_rectangular_region_has_the_boundary_its_corners_imply(
    finder, mini_ccf, acronym
):
    boundaries = finder.region_boundaries(region_acronyms=[acronym])
    contour = boundaries[acronym]

    (x0, x1), (y0, y1) = _bounds(mini_ccf, acronym)
    assert contour[:, 0].min() == pytest.approx(x0)
    assert contour[:, 0].max() == pytest.approx(x1)
    assert contour[:, 1].min() == pytest.approx(y0)
    assert contour[:, 1].max() == pytest.approx(y1)


def test_the_contour_is_a_closed_loop(finder):
    contour = finder.region_boundaries(region_acronyms=["AAA"])["AAA"]
    assert np.array_equal(contour[0], contour[-1])


def test_the_two_regions_have_different_boundaries(finder, mini_ccf):
    """They differ in both size and position, so an axis mixup is caught."""
    boundaries = finder.region_boundaries()
    aaa, bbb = boundaries["AAA"], boundaries["BBB"]

    assert aaa.shape != bbb.shape or not np.array_equal(aaa, bbb)
    assert aaa[:, 0].max() < bbb[:, 0].min()


def test_a_region_absent_from_the_atlas_gives_an_empty_boundary(
    finder, mini_ccf
):
    """Documented: regions specified but not present return empty lists."""
    boundaries = finder.region_boundaries(
        region_acronyms=[mini_ccf.absent_region_acronym]
    )

    assert boundaries[mini_ccf.absent_region_acronym].size == 0


def test_right_hemisphere_reflects_the_boundary(finder, mini_ccf):
    left = finder.region_boundaries(region_acronyms=["AAA"], hemisphere="left")["AAA"]
    right = finder.region_boundaries(region_acronyms=["AAA"], hemisphere="right")["AAA"]

    max_x = mini_ccf.atlas.shape[0]
    assert np.allclose(right[:, 0], max_x - left[:, 0])
    assert np.allclose(right[:, 1], left[:, 1])


def test_right_for_both_shifts_into_the_second_half(finder, mini_ccf):
    left = finder.region_boundaries(region_acronyms=["AAA"], hemisphere="left")["AAA"]
    shifted = finder.region_boundaries(
        region_acronyms=["AAA"], hemisphere="right_for_both"
    )["AAA"]

    max_x = mini_ccf.atlas.shape[0]
    assert np.allclose(shifted[:, 0], 2 * max_x - left[:, 0])


def test_view_space_moves_the_reflection_axis(finder, mini_ccf):
    left = finder.region_boundaries(region_acronyms=["AAA"], hemisphere="left")["AAA"]
    right = finder.region_boundaries(
        region_acronyms=["AAA"], hemisphere="right", view_space_for_other_hemisphere=2
    )["AAA"]

    max_x = mini_ccf.atlas.shape[0] - 2
    assert np.allclose(right[:, 0], max_x - left[:, 0])


# -- region_masks ----------------------------------------------------------


def test_masks_with_default_arguments_are_the_full_view(finder, mini_ccf):
    """The most common call must return actual masks, not empty arrays."""
    masks = finder.region_masks()

    assert set(masks) == set(mini_ccf.atlas_regions)
    for acronym, mask in masks.items():
        assert mask.shape == mini_ccf.atlas.shape
        assert mask.any()


@pytest.mark.parametrize("acronym", ["AAA", "BBB"])
def test_a_mask_marks_exactly_the_regions_pixels(finder, mini_ccf, acronym):
    index = list(mini_ccf.label_acronyms).index(acronym) + 1
    masks = finder.region_masks(region_acronyms=[acronym])

    assert np.array_equal(masks[acronym], mini_ccf.atlas == index)


def test_masks_are_trimmed_when_view_space_is_requested(finder, mini_ccf):
    """AAA sits in the retained half and survives; BBB is trimmed away."""
    trim = mini_ccf.atlas.shape[0] // 2
    masks = finder.region_masks(view_space_for_other_hemisphere=True)

    for mask in masks.values():
        assert mask.shape == (mini_ccf.atlas.shape[0] - trim, mini_ccf.atlas.shape[1])
    assert masks["AAA"].any()
    assert not masks["BBB"].any()


def test_right_hemisphere_masks_are_flipped(finder, mini_ccf):
    left = finder.region_masks(region_acronyms=["AAA"], hemisphere="left")["AAA"]
    right = finder.region_masks(region_acronyms=["AAA"], hemisphere="right")["AAA"]

    assert np.array_equal(right, left[::-1, :])


@pytest.mark.parametrize("hemisphere", ["left_for_both", "right_for_both"])
def test_for_both_masks_are_doubled_in_the_first_dimension(
    finder, mini_ccf, hemisphere
):
    masks = finder.region_masks(region_acronyms=["AAA"], hemisphere=hemisphere)
    mask = masks["AAA"]

    assert mask.shape == (mini_ccf.atlas.shape[0] * 2, mini_ccf.atlas.shape[1])
    half = mini_ccf.atlas.shape[0]
    if hemisphere == "left_for_both":
        assert mask[:half].any() and not mask[half:].any()
    else:
        assert mask[half:].any() and not mask[:half].any()


# -- documented error conditions -------------------------------------------


def test_an_unknown_acronym_raises(finder):
    with pytest.raises(ValueError, match="does not have an index"):
        finder.region_boundaries(region_acronyms=["ZZZ"])


def test_an_unknown_acronym_raises_for_masks(finder):
    with pytest.raises(ValueError, match="does not have an index"):
        finder.region_masks(region_acronyms=["ZZZ"])


@pytest.mark.parametrize("hemisphere", ["middle", "both", "Left"])
def test_an_invalid_hemisphere_raises(finder, hemisphere):
    with pytest.raises(ValueError, match="must be left, right"):
        finder.region_boundaries(hemisphere=hemisphere)


def test_an_unknown_named_view_raises(finder):
    with pytest.raises(ValueError, match="unknown string option"):
        finder.region_boundaries(view_space_for_other_hemisphere="flatmap_buttrfly")


# -- defects ---------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=ABSENT_REGION_REFLECTION)
@pytest.mark.parametrize("hemisphere", ["right", "right_for_both"])
def test_reflecting_a_region_absent_from_the_atlas_degrades_predictably(
    finder, mini_ccf, hemisphere
):
    """An absent region is stored as ``np.array([])``, which is 1-D. The
    reflection step then does ``boundaries[k][:, 0]`` on it and raises
    ``IndexError: too many indices``.

    The same call with ``hemisphere="left"`` works, so asking for the other
    hemisphere turns a documented empty result into a crash.
    """
    boundaries = finder.region_boundaries(
        region_acronyms=[mini_ccf.absent_region_acronym], hemisphere=hemisphere
    )

    assert boundaries[mini_ccf.absent_region_acronym].size == 0


@pytest.mark.xfail(strict=True, reason=ABSENT_REGION_REFLECTION)
def test_a_present_and_an_absent_region_can_be_requested_together(
    finder, mini_ccf
):
    """The realistic form of the defect: one missing region in a list of many
    takes the whole call down."""
    boundaries = finder.region_boundaries(
        region_acronyms=["AAA", mini_ccf.absent_region_acronym], hemisphere="right"
    )

    assert boundaries["AAA"].size > 0
    assert boundaries[mini_ccf.absent_region_acronym].size == 0
