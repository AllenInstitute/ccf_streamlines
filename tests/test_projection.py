import nrrd
import numpy as np
import pytest

from ccf_streamlines.projection import BoundaryFinder


@pytest.fixture
def boundary_finder(tmp_path):
    """A BoundaryFinder over a tiny two-region atlas.

    Region 1 occupies the top two rows, region 2 the bottom two; the middle
    rows are background (0), which `_validate_inputs` requires in order to
    infer the region list.
    """
    atlas = np.zeros((8, 6), dtype=np.uint32)
    atlas[:2, :] = 1
    atlas[-2:, :] = 2

    atlas_file = tmp_path / "atlas.nrrd"
    nrrd.write(str(atlas_file), atlas)

    # Deliberately ragged whitespace: the labels file is parsed with a `\s+`
    # separator, so alignment padding and tabs must be tolerated.
    labels_file = tmp_path / "labels.txt"
    labels_file.write_text("1   255 0   0   1 1 1\tAAA\n2     0 255 0   1 1 1  BBB\n")

    return BoundaryFinder(str(atlas_file), str(labels_file)), atlas


def test_labels_parsed_from_ragged_whitespace(boundary_finder):
    bf, _ = boundary_finder
    assert bf.labels_df["acronym"].tolist() == ["AAA", "BBB"]


def test_region_masks_default_keeps_full_view(boundary_finder):
    # Regression: `view_space_for_other_hemisphere` defaults to False, which
    # validates to 0. Trimming unconditionally made this `raster[:-0, :]`,
    # an empty slice, so every mask came back with shape (0, 6).
    bf, atlas = boundary_finder
    masks = bf.region_masks()

    assert set(masks) == {"AAA", "BBB"}
    for mask in masks.values():
        assert mask.shape == atlas.shape

    assert np.array_equal(masks["AAA"], atlas == 1)
    assert np.array_equal(masks["BBB"], atlas == 2)


def test_region_masks_trims_when_space_requested(boundary_finder):
    bf, atlas = boundary_finder
    masks = bf.region_masks(view_space_for_other_hemisphere=2)

    for mask in masks.values():
        assert mask.shape == (atlas.shape[0] - 2, atlas.shape[1])

    # Region 2 lived in the trimmed-away rows, so nothing is left of it.
    assert not masks["BBB"].any()
    assert np.array_equal(masks["AAA"], (atlas == 1)[:-2, :])


def test_region_masks_right_hemisphere_is_flipped(boundary_finder):
    bf, atlas = boundary_finder
    masks = bf.region_masks(hemisphere="right")

    assert np.array_equal(masks["AAA"], (atlas == 1)[::-1, :])


def test_region_boundaries_default_matches_full_view(boundary_finder):
    bf, atlas = boundary_finder
    boundaries = bf.region_boundaries()

    assert set(boundaries) == {"AAA", "BBB"}
    for contour in boundaries.values():
        assert contour.size > 0
        assert contour[:, 0].max() <= atlas.shape[0]
