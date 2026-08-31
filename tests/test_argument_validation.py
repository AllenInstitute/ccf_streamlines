"""Unrecognised `kind` and `scale` arguments raise, rather than fail silently.

Pins AllenInstitute/ccf_streamlines#21. Before the fix, a typo'd `kind` gave an
all-zeros view from ``Isocortex2dProjector`` -- indistinguishable from a
legitimately empty projection -- and an ``UnboundLocalError`` naming a local
variable from ``IsocortexEntireProjector``; a typo'd `scale` made
``top_of_streamline_coords`` return ``None``, so the caller failed somewhere
unrelated.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import (
    Isocortex2dProjector,
    IsocortexEntireProjector,
)


@pytest.fixture
def projector_2d(mini_ccf):
    return Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, hemisphere="left"
    )


@pytest.fixture
def entire_projector(mini_ccf):
    return IsocortexEntireProjector(
        mini_ccf.surface_paths_file, resolution=mini_ccf.resolution
    )


@pytest.fixture
def ramp_volume(mini_ccf):
    """A volume holding 1..8 down streamline 0 and nothing else."""
    volume = mini_ccf.volume()
    for step, voxel in enumerate(mini_ccf.path_voxels(0)):
        volume[tuple(voxel)] = float(step + 1)
    return volume


def test_2d_projector_rejects_an_unrecognised_kind(projector_2d, ramp_volume):
    with pytest.raises(ValueError, match="`kind`"):
        projector_2d.project_volume(ramp_volume, kind="maximum")


def test_entire_projector_rejects_an_unrecognised_kind(entire_projector, ramp_volume):
    with pytest.raises(ValueError, match="`kind`"):
        entire_projector.project_volume(ramp_volume, kind="maximum")


def test_top_of_streamline_coords_rejects_an_unrecognised_scale(entire_projector):
    with pytest.raises(ValueError, match="`scale`"):
        entire_projector.top_of_streamline_coords(scale="mm")


def test_the_shape_check_still_runs_for_a_valid_kind(entire_projector):
    """`kind` is validated first, so a bad volume must still report the shape."""
    with pytest.raises(ValueError, match="match lookup volume shape"):
        entire_projector.project_volume(np.zeros((2, 2, 2)), kind="max")


@pytest.mark.parametrize("kind", ["max", "min", "mean", "average", "sum"])
def test_every_documented_kind_is_still_accepted(
    projector_2d, entire_projector, ramp_volume, kind
):
    projector_2d.project_volume(ramp_volume.copy(), kind=kind)
    entire_projector.project_volume(ramp_volume.copy(), kind=kind)


@pytest.mark.parametrize("scale", ["voxels", "microns"])
def test_every_documented_scale_is_still_accepted(entire_projector, scale):
    coords = entire_projector.top_of_streamline_coords(scale=scale)
    assert coords.shape[1] == 3
