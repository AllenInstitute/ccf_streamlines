import numpy as np

from ccf_streamlines.projection import ISOCORTEX_LAYER_KEYS

# Structure set IDs from the mouse ontology, in the same pia-to-white-matter
# order as ISOCORTEX_LAYER_KEYS. The names come from that one list so a
# thickness file written here is always keyed the way the projector classes
# read it back.
ISOCORTEX_LAYER_STRUCTURE_SET_IDS = [
    667481440,
    667481441,
    667481445,
    667481446,
    667481449,
    667481450,
]

LAYER_LABELS = dict(zip(ISOCORTEX_LAYER_STRUCTURE_SET_IDS, ISOCORTEX_LAYER_KEYS))


def measure_streamline_layer_thicknesses(layer_volume, paths, resolution):
    """Measure the start, end, and thickness of layers

    Parameters
    ----------
    layer_volume : array
        3D volume with annotated layers
    paths : array
        Streamline paths
    resolution : tuple
        3-tuple of voxel size in x, y, z

    Returns
    -------
    thicknesses : dict
        Dictionary keyed on layers with start, end, and thickness of layers
    """

    # Remove duplicate consecutive voxels from paths
    fixed_paths = np.zeros_like(paths)
    paths_diff = np.diff(paths, axis=1)
    for i in range(paths.shape[0]):
        unique_inds = np.flatnonzero(paths_diff[i, :])
        fixed_paths[i, : len(unique_inds)] = paths[i, :][unique_inds]
    paths = fixed_paths

    max_nonzero_path_inds = np.count_nonzero(paths, axis=1)

    # get voxel coordinates for all path voxels
    path_x, path_y, path_z = np.unravel_index(paths, layer_volume.shape)
    path_voxels = np.stack(
        [path_x * resolution[0], path_y * resolution[1], path_z * resolution[2]]
    )

    # Find distances between consecutive voxels
    deltas = np.diff(path_voxels, axis=2)
    distances = np.sqrt((deltas**2).sum(axis=0))

    # add thickness of last voxel to end of paths
    distances[np.arange(distances.shape[0]), max_nonzero_path_inds - 1] = np.mean(
        resolution
    )

    # Calculate cumulative depths
    cumul_distances = np.cumsum(distances, axis=1)

    layer_annot = layer_volume.flat[paths]

    # Sort the layer annotations so that layers cannot be intercalated
    # (Obviously this is a real hack, but don't think there's a robust
    # solution to the issue)
    # Set zeros to large value so they end up last
    layer_annot[layer_annot == 0] = 999999999
    layer_annot = np.sort(layer_annot, axis=1)

    thicknesses = {}
    end_depths = {}
    start_depths = {}
    for k, v in LAYER_LABELS.items():
        layer_mask = layer_annot == k

        # Last row dropped from np.diff
        layer_mask = layer_mask[:, :-1]

        thicknesses[v] = np.sum(np.where(layer_mask, distances, 0), axis=1)
        end_depths[v] = np.max(np.where(layer_mask, cumul_distances, 0), axis=1)
        start_depths[v] = end_depths[v] - thicknesses[v]

    output = {}
    for k in thicknesses:
        output[k] = np.vstack([start_depths[k], end_depths[k], thicknesses[k]]).T

        # set very small values to exactly zero
        output[k][np.isclose(output[k], 0)] = 0
    return output
