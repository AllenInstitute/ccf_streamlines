Glossary
========

Several words in this project are overloaded, and a few mean different things
in the documentation, the code, and the data files. This page fixes the
meanings so a disagreement about a term can be settled by a document rather
than by a thread.

.. glossary::
    :sorted:

    streamline
        A chain of voxels running from the pia to the white matter at one point
        on the cortical surface, precomputed by solving Laplace's equation
        across isocortex. There are 1,476,024 of them at 10 micron resolution.
        A streamline is identified by its index into the ``paths`` dataset, and
        is *ordered*: element 0 is the pia end.

    path
        Always a :term:`streamline` -- never a file-system path, and never a
        2-D trajectory. Every identifier in the package that contains "path"
        means streamline: ``paths``, ``path_ind``, ``path_ordering``,
        ``matching_path``, ``path_thicknesses``, ``max_path_length``.

        Where a parameter names a file it is the ``_file`` suffix that says so,
        not the word "path": ``surface_paths_file`` is *the file of surface
        paths*, i.e. the file holding the streamlines. The one parameter that
        breaks the suffix convention, ``surface_paths`` in
        :func:`~ccf_streamlines.angle.find_closest_streamline`, accepts either
        a filename or an open HDF5 handle -- but it too is named for the
        streamlines it yields, not for a location on disk.

    flat index
        A single integer identifying a voxel, obtained by
        ``np.ravel_multi_index`` against the volume shape. Both the
        :term:`streamline` array and both lookup tables store flat indices, not
        coordinate triples.

        **Flat index 0 is reserved.** It doubles as the right-padding value in
        ``paths``, so ``paths > 0`` is the validity mask everywhere, and
        ``project_volume`` overwrites ``volume.flat[0]`` with a sentinel before
        a max or min reduction.

    surface voxel
        The first voxel of a :term:`streamline` - the one at the pia. The
        second column of a :term:`view lookup` always holds a surface voxel,
        and the ``volume lookup flat`` dataset is populated only at surface
        voxels.

    volume lookup
        Ambiguous; prefer one of:

        ``volume lookup flat``
            The current dataset in ``surface_paths_10_v3.h5``: a flat array
            mapping a :term:`surface voxel`'s flat index to its
            :term:`streamline` index, holding -1 elsewhere.

        ``volume lookup``
            A 3-D, CCF-shaped dataset in the *superseded* surface-paths files.
            It no longer exists in the current assets.

    closest surface voxel
        A lookup mapping any voxel inside isocortex to the
        :term:`surface voxel` whose streamline best matches it. This is the
        step that lets an arbitrary coordinate find a streamline. A voxel
        missing from it is outside isocortex.

    view
        A 2-D image of isocortex: one of ``top``, ``bottom``, ``back``,
        ``front``, ``medial``, ``side``, ``rotated``, ``flatmap_dorsal``,
        ``flatmap_butterfly``. "View" refers to the *geometry*; the file
        encoding it is a :term:`view lookup`.

    view lookup
        An ``(n, 2)`` array pairing a flat index into the 2-D :term:`view`
        (column 0) with the flat index of the :term:`surface voxel` seen at
        that pixel (column 1). One row per visible pixel.

    tied key
        Two or more :term:`view lookup` rows sharing the same column-1 value,
        i.e. several 2-D pixels showing the same :term:`surface voxel`. Present
        only in the ``flatmap_dorsal``, ``flatmap_butterfly`` and ``rotated``
        views; the other six have none. Ties are why sort stability matters.

    slab
        The 3-D output of :class:`~ccf_streamlines.projection.Isocortex3dProjector`:
        a :term:`view` with a depth axis, so shape ``view size + (depth,)``.
        Distinct from the input CCF volume, which is also 3-D.

    depth
        Distance from the pia along a :term:`streamline`. Reported in voxels by
        default and in microns on request, and rescaled differently by each
        :term:`thickness type` - so "depth" alone is under-specified. Say which
        scale and which thickness type.

    thickness type
        Which normalization a depth or :term:`slab` uses:

        ``unnormalized``
            Distance along the streamline as it actually is, so thickness
            varies across the projection.

        ``normalized_full``
            Pia-to-white-matter distance rescaled to a constant, so overall
            thickness is uniform but layer thicknesses still vary.

        ``normalized_layers``
            Each layer rescaled to a constant thickness, taken from the
            caller's ``layer_thicknesses``. Layers absent from a streamline
            are left as gaps.

    layer thicknesses
        Overloaded, and the two senses are easy to mix up:

        reference (``layer_thicknesses`` argument)
            A dict of *target* thicknesses in microns, one per layer, the same
            for the whole cortex. Usually derived from ``avg_layer_depths.json``.

        per-streamline (``streamline_layer_thickness_file``)
            The *measured* start, end and thickness of each layer for each
            individual streamline, from ``cortical_layers_10_v2.h5``.

        Only the ratios of the reference thicknesses affect a
        ``normalized_layers`` result; the absolute total is discarded when the
        result is rescaled to the padded streamline length.

    hemisphere
        Overloaded across three classes, with different accepted values:

        ``Isocortex2dProjector`` / ``Isocortex3dProjector``
            ``both``, ``left``, ``right``.

        ``IsocortexCoordinateProjector``
            ``both``, ``both_mirrored``, ``left``, ``right``.

        ``BoundaryFinder``
            ``left``, ``left_for_both``, ``right``, ``right_for_both``.

        The reference data covers the left hemisphere only; everything else is
        produced by reflecting across the midline at z = 570.

    view space for other hemisphere
        The number of columns trimmed from the right edge of a :term:`view`
        before two hemispheres are concatenated. Accepted as ``False`` (none),
        ``True`` (half the view), a preset view name, or an explicit integer.

    padded length
        The second dimension of the ``paths`` array - 200 in the real assets.
        Every :term:`streamline` is stored right-padded with zeros to this
        width, and it is also the depth of a normalized :term:`slab`.
