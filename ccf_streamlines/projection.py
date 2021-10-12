import numpy as np
import pandas as pd
import nrrd
import h5py
import logging
from ccf_streamlines.coordinates import coordinates_to_voxels
from skimage.measure import find_contours
from tqdm import tqdm


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

    def project_coordinates(self, coords):
        """ Project set of coordinates to the 2D view

        Accuracy is at the voxel level.

        Parameters
        ----------
        coords : array
            3D spatial coordinates, in microns

        Returns
        -------
        projected_coords : array
            2D projected coordinates, in voxels
        """
        if self.closest_surface_voxel_reference_file is None:
            raise ValueError("Must specific closest surface reference file to project coordinates")

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
        projected_ind = np.zeros_like(matching_surface_voxel_ind)
        for i in range(projected_ind.shape[0]):
            projected_ind[i] = self.view_lookup[
                self.view_lookup[:, 1] == matching_surface_voxel_ind[i], 0][0]

        # Convert the flattened indices to 2D coordinates
        projected_coords = np.unravel_index(
            projected_ind,
            self.view_size
        )
        return projected_coords


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

        for i in tqdm(range(self.paths.shape[0])):
            # Get the coordinates for the path on the surface of the slab
            r, c = np.unravel_index(self.view_lookup[i, 0], self.view_size)

            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r, c, :] = volume.flat[self.paths[i, :]]
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

        for i in tqdm(range(self.paths.shape[0])):
            # Get the coordinates for the path on the surface of the slab
            r, c = np.unravel_index(self.view_lookup[i, 0], self.view_size)

            # Get the path data (dropping the zeros)
            path_ind = self.paths[i, :]
            path_ind = path_ind[path_ind > 0]

            # Interpolate the path data to the full thickness
            path_thickness = len(path_ind)
            interp_vol = np.interp(
                np.linspace(0, path_thickness, full_thickness),
                range(path_thickness),
                volume.flat[path_ind]
            )

            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r, c, :] = interp_vol
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
        full_thickness_voxels = self.paths.shape[1]
        ref_full_thickness = np.sum(list(self.layer_thicknesses.values()))
        ref_thickness_voxels = {k: int(np.round(full_thickness_voxels * t / ref_full_thickness))
            for k, t in self.layer_thicknesses.items()}

        for i in tqdm(range(self.paths.shape[0])):
            # Get the path data (dropping the zeros)
            path_ind = self.paths[i, :]
            path_ind = path_ind[path_ind > 0]

            # Create interpolation lookup
            interp_list = []
            path_thickness = 0
            for k in self.ISOCORTEX_LAYER_KEYS:
                pl_start, pl_end, pl_thick = self.path_layer_thickness[k][i, :]
                if pl_start == 0 and pl_end == 0:
                    # if layer not present in streamline, set to -1
                    interp_list.append(-1 * np.ones(ref_thickness_voxels[k]))
                else:
                    path_thickness += pl_thick
                    interp_list.append(
                        np.linspace(pl_start, pl_end, ref_thickness_voxels[k])
                    )
            interp_vals = np.concatenate(interp_list)

            # Interpolate the path data to the specified layer thicknesses
            interp_vol = np.interp(
                interp_vals,
                np.linspace(0, path_thickness, len(path_ind)),
                volume.flat[path_ind],
                left=0 # set values of -1 to 0 (i.e. empty for layers not present)
            )

            # Get the coordinates for the path on the surface of the slab
            r, c = np.unravel_index(self.view_lookup[i, 0], self.view_size)

            # Fill in the thickness of the slab at that location from the volume
            projected_volume[r, c, :] = interp_vol
        return projected_volume

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