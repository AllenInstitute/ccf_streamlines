import numpy as np
import pandas as pd
import nrrd
import h5py
import logging
from ccf_streamlines.coordinates import coordinates_to_voxels
from skimage.measure import find_contours
from tqdm import tqdm
from scipy.spatial.distance import cdist


class Isocortex2dProjector:
    """ 2D projection of the common cortical framework

    Parameters
    ----------
    projection_file : str
        File path to an HDF5 file containing the 2D projection information.
    surface_paths_file : str
        File path to an HDF5 file containing information about the paths between
        the top and bottom of cortex.
    closest_surface_voxel_reference_file : str, optional
        File path to a NRRD file containing information about the closest
        streamlines for voxels within the isocortex.
    single_hemisphere : bool, default True
        Whether to collapse data into a single hemisphere visualization
    """

    def __init__(self,
        projection_file,
        surface_paths_file,
        closest_surface_voxel_reference_file=None,
        single_hemisphere=True,
    ):
        self.single_hemisphere = single_hemisphere

        # Load the projection information
        logging.info("Loading projection file")
        with h5py.File(projection_file, "r") as proj_f:
            self.view_lookup = proj_f["view lookup"][:]
            self.view_size = proj_f.attrs["view size"][:]
            self.resolution = tuple(int(d.decode()) for d in proj_f.attrs["spacing"][:])

        # Load the surface path information
        logging.info("Loading surface path file")
        with h5py.File(surface_paths_file, "r") as path_f:
            self.paths = path_f["paths"][:]
            self.volume_lookup = path_f["volume lookup"][:]

        # Remove duplicate consecutive voxels from the paths
        fixed_paths = np.zeros_like(self.paths)
        paths_diff = np.diff(self.paths, axis=1)
        for i in range(self.paths.shape[0]):
            unique_inds = np.flatnonzero(paths_diff[i, :])
            fixed_paths[i, :len(unique_inds)] = self.paths[i, :][unique_inds]
        self.paths = fixed_paths

        # Select and order paths to match the projection.
        # The view_lookup array contains the indices of the 2D view in the first
        # column and indices of the (flattened) 3D volume in the second.
        # We find the indices of the paths by going to the appropriate voxels
        # in volume_lookup.
        self.paths = self.paths[
            self.volume_lookup.flat[self.view_lookup[:, 1]],
            :
        ]

        # Load the closest surface voxel reference file, if provided
        if closest_surface_voxel_reference_file is not None:
            logging.info("Loading closest surface reference file")
            self.closest_surface_voxels, _ = nrrd.read(
                closest_surface_voxel_reference_file)
        else:
            self.closest_surface_voxels = None


    def project_volume(self, volume, kind="max"):
        """ Create a maximum projection view of the volume

        Parameters
        ----------
        volume : array
            Input volume with size matching the lookup volume
        kind : {'max', 'min', 'mean'}
            Whether to create a minimum, maximum, or mean projection

        Returns
        -------
        projected_volume : array
            2D projection of input volume
        """
        if volume.shape != self.volume_lookup.shape:
            raise ValueError(
                f"Input volume must match lookup volume shape; {volume.shape} != {self.volume_lookup.shape}")

        projected_volume = np.zeros(self.view_size, dtype=volume.dtype)

        if kind == "max":
            # The path specification assumes the first point in the volume is not a
            # valid data point and so should be ignored. Since we are doing a
            # maximum projection, we set that to the minimum possible value
            # so that it won't be selected
            volume.flat[0] = np.iinfo(volume.dtype).min
            for i in tqdm(range(self.paths.shape[0])):
                projected_volume.flat[self.view_lookup[i, 0]] = np.max(
                    volume.flat[self.paths[i, :]])
        elif kind == "min":
            # Same thing as above, just set to maximum instead of minimum
            volume.flat[0] = np.iinfo(volume.dtype).max
            for i in tqdm(range(self.paths.shape[0])):
                projected_volume.flat[self.view_lookup[i, 0]] = np.min(
                    volume.flat[self.paths[i, :]])
        elif kind == "mean":
            # Don't use paths parts with a value of zero (can't use the
            # simplifying trick above)
            for i in tqdm(range(self.paths.shape[0])):
                path_ind = self.paths[i, :][self.paths[i, :] > 0]
                projected_volume.flat[self.view_lookup[i, 0]] = np.mean(
                    volume.flat[path_ind])
        return projected_volume

    def project_coordinates(self, coords, scale="voxels"):
        """ Project set of coordinates to the 2D view

        Accuracy is at the voxel level.

        Parameters
        ----------
        coords : array
            3D spatial coordinates, in microns
        scale : {"voxels", "microns"}
            Scale for projected coordinates. For ease of overlay on projected
            images, use "voxels". For actual distances, use "microns".

        Returns
        -------
        projected_coords : array
            2D projected coordinates, in voxels or microns (depending on `scale`)
        """
        if scale not in {"voxels", "microns"}:
            raise ValueError(f"`scale` must be either 'voxels' or 'microns'; was {scale}")

        projected_coords, _, _ = self._get_projected_coordinates_and_surface_voxels(coords)
        if scale == "microns":
            return projected_coords[0] * self.resolution[0], projected_coords[1] * self.resolution[1]
        else:
            return projected_coords

    def _get_projected_coordinates_and_surface_voxels(self, coords):
        if self.closest_surface_voxels is None:
            raise ValueError("Must specify closest surface reference file to project coordinates")

        # Find the voxels containing the coordinates
        voxels = coordinates_to_voxels(coords, resolution=self.resolution)

        if self.single_hemisphere:
            # Reflect voxels in other hemisphere to projected hemisphere.
            # Projected hemisphere is in lower half of z-dimension
            z_size = self.closest_surface_voxels.shape[2]
            z_midline = z_size / 2
            voxels[voxels[:, 2] > z_midline, 2] = z_size - voxels[voxels[:, 2] > z_midline, 2]

        # Find the surface voxels that best match the voxels
        voxel_ind = np.ravel_multi_index(
            tuple(voxels[:, i] for i in range(voxels.shape[1])),
            self.closest_surface_voxels.shape
        )
        matching_surface_voxel_ind = self.closest_surface_voxels.flat[voxel_ind]

        # Find the flattened projection indices for those surface voxels
        projected_ind = np.zeros_like(matching_surface_voxel_ind, dtype=int)
        for i in range(projected_ind.shape[0]):
            matching_lookups = self.view_lookup[
                self.view_lookup[:, 1] == matching_surface_voxel_ind[i], 0]
            if len(matching_lookups) == 0:
                # cannot not find location for this coordinate
                # use sentinel value of -1 to indicate that it's missing
                projected_ind[i] = -1
            else:
                projected_ind[i] = matching_lookups[0]

        # Convert the flattened indices to 2D coordinates
        projected_coords_not_missing = np.unravel_index(
            projected_ind[projected_ind != -1],
            self.view_size
        )

        projected_coords_x = np.zeros_like(projected_ind, dtype=float)
        projected_coords_y = np.zeros_like(projected_ind, dtype=float)
        projected_coords_x[projected_ind != -1] = projected_coords_not_missing[0]
        projected_coords_x[projected_ind == -1] = np.nan
        projected_coords_y[projected_ind != -1] = projected_coords_not_missing[1]
        projected_coords_y[projected_ind == -1] = np.nan

        return (
            (projected_coords_x, projected_coords_y),
            voxels,
            matching_surface_voxel_ind,
        )

    def streamline_for_voxel(self, voxel):
        """ Find the streamline that is closest to the specified voxel

        Parameters
        ----------
        voxel : array
            3D coordinates (integer values) of voxel

        Returns
        -------
        streamline : array
            Array of flattened indices of the voxels belonging to the streamline
        """
        voxel_ind = np.ravel_multi_index(
            tuple(voxel),
            self.closest_surface_voxels.shape
        )
        matching_surface_voxel_ind = self.closest_surface_voxels.flat[voxel_ind]
        path_ind = np.flatnonzero(self.view_lookup[:, 1] == matching_surface_voxel_ind)
        if len(path_ind) == 0:
            # cannot not find location for this coordinate
            # use sentinel value of -1 to indicate that it's missing
            logging.warning("No streamline found for voxel")
            return None
        else:
            path_ind = path_ind[0]
        return self.paths[path_ind, :]

    def project_path_ordered_data(self, data):
        """ Project 1D data corresponding to the list of streamlines

        Parameters
        ----------
        data : array
            1D array of data corresponding to streamlines

        Returns
        -------
        projection : array
            2D projection of data
        """
        ordered_data = data[
            self.volume_lookup.flat[self.view_lookup[:, 1]]
        ]

        projection = np.zeros(self.view_size, dtype=data.dtype)
        projection.flat[self.view_lookup[:, 0]] = ordered_data

        return projection


class Isocortex3dProjector(Isocortex2dProjector):
    """ Flattened projection of the common cortical framework with thickness

    Parameters
    ----------
    projection_file : str
        File path to an HDF5 file containing the 2D projection information.
    surface_paths_file : str
        File path to an HDF5 file containing information about the paths between
        the top and bottom of cortex.
    thickness_type : {"unnormalized", "normalized_full", "normalized_layers"}
        Type of thickness normalization. If 'unnormalized', the thickness
        varies across the projection according to the length of the streamline.
        If 'normalized_full', the thickness (between pia and white matter) is
        consistent everywhere, but layer thicknesses can vary. If
        'normalized_layers', layer thicknesses are consistent everywhere. Layers
        that are not present in a particular region are left empty in the
        projection.
    layer_thicknesses : dict, optional
        Default None. Dictionary of layer thicknesses. Only used if
        `thickness_type` is `normalized_layers`.
    streamline_layer_thickness_file : str, optional
        Default None. File path to an HDF5 file containing information about
        the layer thicknesses per streamline.
    closest_surface_voxel_reference_file : str, optional
        File path to a NRRD file containing information about the closest
        streamlines for voxels within the isocortex.
    single_hemisphere : bool, default True
        Whether to collapse data into a single hemisphere visualization
    """
    ISOCORTEX_LAYER_KEYS = [
        'Isocortex layer 1',
        'Isocortex layer 2/3', # hilariously, this goes into a group in the h5 file
        'Isocortex layer 4',
        'Isocortex layer 5',
        'Isocortex layer 6a',
        'Isocortex layer 6b'
    ]

    def __init__(self,
        projection_file,
        surface_paths_file,
        thickness_type="unnormalized",
        layer_thicknesses=None,
        streamline_layer_thickness_file=None,
        closest_surface_voxel_reference_file=None,
        single_hemisphere=True,
    ):
        super().__init__(
            projection_file,
            surface_paths_file,
            closest_surface_voxel_reference_file,
            single_hemisphere
        )

        allowed_thickness_types = {"unnormalized", "normalized_full", "normalized_layers"}
        if thickness_type not in allowed_thickness_types:
            raise ValueError(f"{thickness_type} not in allowed values {allowed_thickness_types}")
        self.thickness_type = thickness_type

        self.layer_thicknesses = layer_thicknesses

        if thickness_type is "normalized_layers":
            if streamline_layer_thickness_file is None:
                raise ValueError("`streamline_layer_thickness_file` cannot be None if `thickness_type` is `normalized_layers`")
            if layer_thicknesses is None:
                raise ValueError("`layer_thicknesses` cannot be None if `thickness_type` is `normalized_layers`")

            self.path_layer_thickness = {}
            with h5py.File(streamline_layer_thickness_file, "r") as f:
                for k in self.ISOCORTEX_LAYER_KEYS:
                    self.path_layer_thickness[k] = f[k][:]

                    # Select and order paths to match the projection.
                    self.path_layer_thickness[k] = self.path_layer_thickness[k][
                        self.volume_lookup.flat[self.view_lookup[:, 1]], :]

    def project_volume(self, volume):
        """ Create a flattened slab view of the volume.

        Parameters
        ----------
        volume : array
            Input volume with size matching the lookup volume

        Returns
        -------
        projected_volume : array
            3D slab projection of input volume
        """
        if volume.shape != self.volume_lookup.shape:
            raise ValueError(
                f"Input volume must match lookup volume shape; {volume.shape} != {self.volume_lookup.shape}")

        if self.thickness_type == "unnormalized":
            return self._project_volume_unnormalized(volume)
        elif self.thickness_type == "normalized_full":
            return self._project_volume_normalized_full(volume)
        elif self.thickness_type == "normalized_layers":
            return self._project_volume_normalized_layers(volume)
        else:
            raise ValueError(f"Unknown thickness type {self.thickness_type}")

    def _project_volume_unnormalized(self, volume):
        """ Create a flattened slab view of the volume with thickness as in the volume.

        Parameters
        ----------
        volume : array
            Input volume with size matching the lookup volume

        Returns
        -------
        projected_volume : array
            3D slab projection of input volume
        """
        projected_volume = np.zeros(
            tuple(self.view_size) + (self.paths.shape[1],),
            dtype=volume.dtype)

        # Get the coordinates for the paths on the surface of the slab
        r, c = np.unravel_index(self.view_lookup[:, 0], self.view_size)
        for i in tqdm(range(self.paths.shape[0])):
            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r[i], c[i], :] = volume.flat[self.paths[i, :]]
        return projected_volume

    def _project_volume_normalized_full(self, volume):
        """ Create a flattened slab view of the volume with equal overall thickness.

        Parameters
        ----------
        volume : array
            Input volume with size matching the lookup volume

        Returns
        -------
        projected_volume : array
            3D slab projection of input volume
        """
        projected_volume = np.zeros(
            tuple(self.view_size) + (self.paths.shape[1],),
            dtype=volume.dtype)
        full_thickness = self.paths.shape[1]

        # Get the coordinates for the paths on the surface of the slab
        r, c = np.unravel_index(self.view_lookup[:, 0], self.view_size)

        # Calculate the thickness of each path
        path_thicknesses = np.count_nonzero(self.paths, axis=1)

        for i in tqdm(range(self.paths.shape[0])):
            # Get the path data (dropping the zeros)
            path_ind = self.paths[i, :path_thicknesses[i]]

            # Interpolate the path data to the full thickness
            interp_vol = np.interp(
                np.linspace(0, path_thicknesses[i], full_thickness),
                np.arange(path_thicknesses[i]),
                volume.flat[path_ind]
            )

            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r[i], c[i], :] = interp_vol
        return projected_volume

    def _project_volume_normalized_layers(self, volume):
        """ Create a flattened slab view of the volume with equal-thickness layers.

        Parameters
        ----------
        volume : array
            Input volume with size matching the lookup volume

        Returns
        -------
        projected_volume : array
            3D slab projection of input volume
        """
        projected_volume = np.zeros(
            tuple(self.view_size) + (self.paths.shape[1],),
            dtype=volume.dtype)

        ref_thickness_voxels = self._get_reference_layer_thicknesses_in_voxels()

        # Get the coordinates for the path on the surface of the slab
        r, c = np.unravel_index(self.view_lookup[:, 0], self.view_size)

        max_nonzero_path_inds = np.count_nonzero(self.paths, axis=1)

        all_layer_thicknesses = np.vstack(
            [self.path_layer_thickness[k][:, 2] for k in self.ISOCORTEX_LAYER_KEYS]
        ).T
        path_thicknesses = np.sum(all_layer_thicknesses, axis=1)

        # TODO - think of ways to vectorize some of these operations
        for i in tqdm(range(self.paths.shape[0])):
            # Get the path data (dropping the zeros)
            path_ind = self.paths[i, :max_nonzero_path_inds[i]]

            # Create interpolation lookup
            interp_list = []
            for k in self.ISOCORTEX_LAYER_KEYS:
                pl_start, pl_end, _ = self.path_layer_thickness[k][i, :]
                if pl_start == 0 and pl_end == 0:
                    # if layer not present in streamline, set to -1
                    interp_list.append(-1 * np.ones(ref_thickness_voxels[k]))
                else:
                    interp_list.append(
                        np.linspace(pl_start, pl_end, ref_thickness_voxels[k])
                    )
            interp_vals = np.concatenate(interp_list)

            # Interpolate the path data to the specified layer thicknesses
            interp_vol = np.interp(
                interp_vals,
                np.linspace(0, path_thicknesses[i], max_nonzero_path_inds[i]),
                volume.flat[path_ind],
                left=0 # set values of -1 to 0 (i.e. empty for layers not present)
            )

            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r[i], c[i], :] = interp_vol
        return projected_volume

    def _get_reference_layer_thicknesses_in_voxels(self):
        full_thickness_voxels = self.paths.shape[1]
        ref_full_thickness = np.sum(list(self.layer_thicknesses.values()))
        ref_thickness_voxels = {k: int(np.round(full_thickness_voxels * t / ref_full_thickness))
            for k, t in self.layer_thicknesses.items()}
        return ref_thickness_voxels

    def project_coordinates(self, coords, scale="voxels"):
        """ Project set of coordinates to the flattened slab

        Accuracy is at the voxel level.

        Parameters
        ----------
        coords : array
            3D spatial coordinates, in microns
        scale : {"voxels", "microns"}
            Scale for projected coordinates. For ease of overlay on projected
            images, use "voxels". For actual distances, use "microns".

        Returns
        -------
        projected_coords : array
            3D projected coordinates
        """
        if scale not in {"voxels", "microns"}:
            raise ValueError(f"`scale` must be either 'voxels' or 'microns'; was {scale}")

        projected_2d_coords, voxels, matching_surface_voxel_ind = self._get_projected_coordinates_and_surface_voxels(coords)

        full_thickness_voxels = self.paths.shape[1]
        depth = []
        for i in range(matching_surface_voxel_ind.shape[0]):
            # Get 3D coordinates of voxels of nearest path

            path_idx = np.flatnonzero(self.view_lookup[:, 1] == matching_surface_voxel_ind[i])
            if len(path_idx) == 0:
                # No matching path in lookup - cannot find depth
                depth.append(np.nan)
                continue
            else:
                path_idx = path_idx[0]
            matching_path = self.paths[path_idx, :]
            matching_path = matching_path[matching_path > 0]
            matching_path_voxels = np.unravel_index(
                matching_path, self.volume_lookup.shape)
            matching_path_voxels = np.array(matching_path_voxels).T

            # Find voxel on path closest to the coordinate's voxel
            dist_to_path = cdist(voxels[i, :][np.newaxis, :], matching_path_voxels)
            min_dist_idx = np.argmin(dist_to_path)

            # Calculate the depth
            if self.thickness_type == "unnormalized":
                depth.append(min_dist_idx)
            elif self.thickness_type == "normalized_full":
                depth.append(min_dist_idx / len(matching_path) * full_thickness_voxels)
            elif self.thickness_type == "normalized_layers":
                frac_along_path = min_dist_idx / len(matching_path)
                # Figure out how long the path is
                path_thickness = 0
                for k in self.ISOCORTEX_LAYER_KEYS:
                    pl_start, pl_end, pl_thick = self.path_layer_thickness[k][path_idx, :]
                    path_thickness += pl_thick
                depth_in_path = frac_along_path * path_thickness

                ref_layer_top = 0
                for k in self.ISOCORTEX_LAYER_KEYS:
                    pl_start, pl_end, pl_thick = self.path_layer_thickness[k][path_idx, :]
                    if pl_start == 0 and pl_end == 0:
                        # Layer not present - skip it
                        continue
                    if depth_in_path <= pl_end:
                        fraction_through_layer = (depth_in_path - pl_start) / (pl_end - pl_start)
                        depth.append(fraction_through_layer * self.layer_thicknesses[k] + ref_layer_top)
                        break
                    else:
                        ref_layer_top += self.layer_thicknesses[k]

        if self.thickness_type == "normalized_layers":
            ref_total_thickness = np.sum(list(self.layer_thicknesses.values()))
            depth = np.array(depth) / ref_total_thickness * full_thickness_voxels
        else:
            depth = np.array(depth)

        if scale == "microns":
            if self.thickness_type == "normalized_layers":
                depth_microns = depth * ref_total_thickness / full_thickness_voxels
            else:
                depth_microns = depth * self.resolution[2]
            return (
                projected_2d_coords[0] * self.resolution[0],
                projected_2d_coords[1] * self.resolution[1],
                depth_microns
            )
        else:
            return projected_2d_coords[0], projected_2d_coords[1], depth


class BoundaryFinder:
    """ Boundaries of cortical regions from 2D atlas projections

    Parameters
    ----------
    projected_atlas_file : str
        File path to a NRRD file containing the 2D projection of the atlas
        labeled consistently with the `labels_file`.
    labels_file : str
        File path to a text file containing the region labels.
    single_hemisphere : bool, default True
        Whether to collapse data into a single hemisphere visualization

    """

    def __init__(self,
        projected_atlas_file,
        labels_file,
        single_hemisphere=True,
    ):
        self.single_hemisphere = single_hemisphere

        # Load the projection
        logging.info("Loading projected atlas file")
        self.proj_atlas, self.proj_atlas_meta = nrrd.read(
            projected_atlas_file)

        # Load the labels
        self.labels_df =  pd.read_csv(
            labels_file,
            header=None,
            sep="\s+",
            index_col=0
        )
        self.labels_df.columns = ["r", "g", "b", "x0", "x1", "x2", "acronym"]

    def region_boundaries(self, region_acronyms=None):
        """Get projection coordinates of region boundaries.

        Parameters
        ----------
        region_acronyms : list, optional
            List of regions whose boundaries will be found. If None (default),
            return all regions found in projection. Regions specified but not
            present will have empty coordinate lists returned.

        Returns
        -------
        boundaries : dict
            Dictionary of region boundary coordinates with region acronyms
            as keys.
        """
        if region_acronyms is None:
            unique_entries = np.unique(self.proj_atlas).tolist()
            unique_entries.remove(0) # 0 is defined as not a structure
            region_acronyms = self.labels_df.loc[unique_entries, "acronym"].tolist()
        else:
            label_acronym_set = set(self.labels_df["acronym"].tolist())
            for acronym in region_acronyms:
                if acronym not in label_acronym_set:
                    raise ValueError(f"Region acronym {acronym} does not have an index")

        boundaries = {}
        for acronym in region_acronyms:
            ind = self.labels_df.index[self.labels_df["acronym"] == acronym][0]
            region_raster = np.zeros_like(self.proj_atlas).astype(int)
            region_raster[self.proj_atlas == ind] = 1
            contours = find_contours(region_raster, level=0.5)

            if len(contours) == 0:
                # No contours found
                boundaries[acronym] = np.array([])
            elif len(contours) == 1:
                boundaries[acronym] = contours[0]
            else:
                # Find the biggest contour
                max_len = 0
                for c in contours:
                    if len(c) > max_len:
                        boundaries[acronym] = c
                        max_len = len(c)
        return boundaries