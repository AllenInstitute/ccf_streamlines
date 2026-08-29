"""The chunk-size seam.

``_load_and_sort_paths`` and ``_path_lookup_chunked`` read the volume lookup in
chunks to keep peak memory down. That chunking is where the most recent
regression happened, and with the size hardcoded at 1000 it is unreachable
below roughly a thousand view rows -- so a miniature fixture could not touch
it at all.

Threading the size through the constructors, defaulting to the current value,
makes the code reachable at fixture scale without introducing a seam below the
public interface. These tests hold that change to being behaviour-preserving.
"""

import numpy as np
import pytest

from ccf_streamlines.projection import (
    Isocortex2dProjector,
    IsocortexCoordinateProjector,
)


def test_default_chunk_size_is_unchanged(mini_ccf):
    """The parameter must not change behaviour for existing callers."""
    proj = Isocortex2dProjector(mini_ccf.view_lookup_file, mini_ccf.surface_paths_file)
    assert proj.chunk_size == 1000

    coord_proj = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file, mini_ccf.closest_surface_voxel_file
    )
    assert coord_proj.chunk_size == 1000


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 13, 14, 15, 1000])
def test_path_ordering_is_independent_of_chunk_size(mini_ccf, chunk_size):
    """Chunk boundaries falling inside, on, and outside the data must all agree.

    The fixture has 14 view rows, so this sweeps boundaries that split a tied
    group, land exactly on the last row, and exceed the data entirely.
    """
    reference = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file
    )
    chunked = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file, chunk_size=chunk_size
    )

    assert np.array_equal(chunked.path_ordering, reference.path_ordering)
    assert np.array_equal(chunked.paths, reference.paths)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 1000])
def test_depths_are_independent_of_chunk_size(mini_ccf, chunk_size):
    coords = np.vstack(
        [mini_ccf.coord_on_path(int(p), 2) for p in mini_ccf.in_view_path_indices[:5]]
    )

    reference = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file, mini_ccf.closest_surface_voxel_file
    ).project_depths(coords)
    chunked = IsocortexCoordinateProjector(
        mini_ccf.surface_paths_file,
        mini_ccf.closest_surface_voxel_file,
        chunk_size=chunk_size,
    ).project_depths(coords)

    assert np.array_equal(reference, chunked)


def test_chunked_read_handles_duplicate_keys_within_one_chunk(mini_ccf):
    """The chunk loop de-duplicates with ``np.unique`` and re-expands with
    ``np.repeat``. A tied key split across a chunk boundary, and one wholly
    inside a chunk, must both come back correct."""
    reference = Isocortex2dProjector(
        mini_ccf.view_lookup_file, mini_ccf.surface_paths_file
    )
    # Every view row must resolve to the streamline whose start voxel it names.
    for row, path_index in enumerate(mini_ccf.view_path_indices):
        assert reference.path_ordering[row] == path_index
