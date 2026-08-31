"""Regression tests for issue #20 -- ``project_volume`` corrupting its input.

``paths`` is zero-padded on the right and flat index 0 is reserved as "no
voxel", so a max/min reduction has to keep the padded entries from being
selected. It used to do that by writing the dtype's extreme value into
``volume.flat[0]`` and never restoring it, which modified the caller's array:
projecting twice gave different answers, and with ``hemisphere="both"`` the
second pass wrote through the ``np.flip`` *view*, clobbering ``volume[0, 0, -1]``
as well.

The substitution now happens in the gathered copy, so these tests assert both
halves of the contract: the caller's volume is untouched, *and* the reserved
voxel is still kept out of the reduction.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import (
    Isocortex2dProjector,
    IsocortexEntireProjector,
)


@pytest.fixture
def projector_factory(mini_ccf):
    """Build an ``Isocortex2dProjector`` over the mini-CCF for a hemisphere."""

    def build(hemisphere="left"):
        return Isocortex2dProjector(
            mini_ccf.view_lookup_file,
            mini_ccf.surface_paths_file,
            hemisphere=hemisphere,
        )

    return build


@pytest.fixture
def entire_projector(mini_ccf):
    return IsocortexEntireProjector(mini_ccf.surface_paths_file)


def ramp_volume(mini, dtype=np.float64):
    """A volume whose every voxel holds a distinct value, smallest at index 0.

    Starting the ramp at 1 leaves ``volume.flat[0]`` as the unique global
    minimum, which is what makes the ``kind="min"`` value tests below able to
    see a leak of the reserved voxel.
    """
    volume = mini.volume(dtype=dtype)
    volume.flat[:] = np.arange(1, volume.size + 1)
    return volume


def path_extreme(mini, volume, path_index, kind):
    """Reduce one streamline's voxels directly, without touching the library."""
    voxels = mini.path_voxels(path_index)
    values = volume[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return values.max() if kind == "max" else values.min()


# ---------------------------------------------------------------------------
# The caller's volume must survive a projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_projection_does_not_modify_the_callers_volume(
    mini_ccf, projector_factory, hemisphere, kind
):
    """No hemisphere branch may write into the input.

    ``"right"`` and ``"both"`` are the interesting ones: they project
    ``np.flip(volume, axis=2)``, a view onto the same buffer, so a write at
    flat index 0 of the flipped view lands at ``volume[0, 0, -1]``.
    """
    projector = projector_factory(hemisphere)
    volume = ramp_volume(mini_ccf)
    before = volume.copy()

    projector.project_volume(volume, kind=kind)

    # Named separately from the whole-array check so a regression says which
    # of the two cells the old code wrote to.
    assert volume.flat[0] == before.flat[0]
    assert volume[0, 0, -1] == before[0, 0, -1]
    assert np.array_equal(volume, before)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint32, np.int16])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_projection_leaves_every_dtype_alone(mini_ccf, projector_factory, dtype, kind):
    """The sentinel came from ``np.iinfo``/``np.finfo``; cover both branches."""
    projector = projector_factory("both")
    volume = ramp_volume(mini_ccf, dtype=dtype)
    before = volume.copy()

    projector.project_volume(volume, kind=kind)

    assert np.array_equal(volume, before)


@pytest.mark.parametrize("hemisphere", ["left", "right", "both"])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_projecting_the_same_volume_twice_gives_the_same_answer(
    mini_ccf, projector_factory, hemisphere, kind
):
    """Characterization: this passed before the fix too, and says why.

    Repeatability is the invariant a caller actually notices, so it is worth
    pinning -- but it is not what detects this defect. The old code wrote the
    same sentinel on every call, and neither cell it clobbered, ``(0, 0, 0)``
    or ``(0, 0, z_size - 1)``, lies on a mini-CCF streamline: the fixture puts
    every path in the interior. The corruption was therefore invisible to a
    repeated projection and only visible in the volume itself, which is what
    the tests above assert.
    """
    projector = projector_factory(hemisphere)
    volume = ramp_volume(mini_ccf)

    first = projector.project_volume(volume, kind=kind)
    second = projector.project_volume(volume, kind=kind)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_a_read_only_volume_can_be_projected(mini_ccf, projector_factory, kind):
    """A memory-mapped or otherwise read-only reference volume must work.

    Not merely a nicety: writing the sentinel raised ``ValueError: assignment
    destination is read-only`` for anything opened read-only.
    """
    projector = projector_factory("both")
    volume = ramp_volume(mini_ccf)
    volume.setflags(write=False)

    projected = projector.project_volume(volume, kind=kind)

    assert projected.shape[0] > 0


@pytest.mark.parametrize("kind", ["max", "min"])
def test_entire_projection_does_not_modify_the_callers_volume(
    mini_ccf, entire_projector, kind
):
    volume = ramp_volume(mini_ccf)
    before = volume.copy()

    entire_projector.project_volume(volume, kind=kind)

    assert np.array_equal(volume, before)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_entire_projection_is_repeatable(mini_ccf, entire_projector, kind):
    """Characterization, for the reason given above."""
    volume = ramp_volume(mini_ccf)

    first = entire_projector.project_volume(volume, kind=kind)
    second = entire_projector.project_volume(volume, kind=kind)

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# ...and the reserved voxel must still be excluded from the reduction
# ---------------------------------------------------------------------------


def test_min_projection_ignores_the_reserved_first_voxel(mini_ccf, projector_factory):
    """The padding must not drag every streamline down to ``volume.flat[0]``.

    Every mini-CCF streamline is padded, and the ramp makes index 0 the unique
    global minimum, so a projection that reduced over the padding would return
    that value for every pixel.
    """
    projector = projector_factory("left")
    volume = ramp_volume(mini_ccf)

    projected = projector.project_volume(volume, kind="min")

    covered = []
    for path_index in mini_ccf.in_view_path_indices:
        row, col = mini_ccf.view_pixel_for_path(int(path_index))
        assert projected[row, col] == path_extreme(
            mini_ccf, volume, int(path_index), "min"
        )
        covered.append(projected[row, col])

    # Pixels no streamline reaches keep the zero the view was built with, so
    # the claim is about the covered ones.
    assert min(covered) > volume.flat[0]


def test_max_projection_ignores_the_reserved_first_voxel(mini_ccf, projector_factory):
    """The mirror image: index 0 is made the unique global maximum."""
    projector = projector_factory("left")
    volume = ramp_volume(mini_ccf)
    volume.flat[0] = volume.size + 1

    projected = projector.project_volume(volume, kind="max")

    for path_index in mini_ccf.in_view_path_indices:
        row, col = mini_ccf.view_pixel_for_path(int(path_index))
        assert projected[row, col] == path_extreme(
            mini_ccf, volume, int(path_index), "max"
        )
    assert projected.max() < volume.flat[0]


@pytest.mark.parametrize("kind", ["max", "min"])
def test_entire_projection_ignores_the_reserved_first_voxel(
    mini_ccf, entire_projector, kind
):
    """Same contract for the every-streamline projector, whose result is 1D."""
    volume = ramp_volume(mini_ccf)
    if kind == "max":
        volume.flat[0] = volume.size + 1

    values = entire_projector.project_volume(volume, kind=kind)

    assert len(values) == mini_ccf.paths.shape[0]
    for path_index in range(mini_ccf.paths.shape[0]):
        assert values[path_index] == path_extreme(mini_ccf, volume, path_index, kind)
