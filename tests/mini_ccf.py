"""A miniature but structurally faithful CCF, built into a temporary directory.

The package's entry points are classes whose constructors read hundreds of
megabytes of HDF5. None of that can live in the repository or on a CI runner,
and an external contributor cannot be expected to obtain it. But the package
makes no network calls and hardcodes no asset paths -- every file is a
caller-supplied argument -- so a small set of files with the *same structure*
exercises the whole package through its real public interface, with no mocking
and no source changes.

This module builds that set: six files totalling roughly 40 KB.

    surface_paths_10_v3.h5      streamline voxel paths + flat voxel->path lookup
    view_lookup.h5              2D view pixel <-> 3D surface voxel correspondence
    cortical_layers_10_v2.h5    per-streamline layer start/end/thickness
    closest_surface_voxel.h5    arbitrary voxel -> nearest surface voxel
    atlas_2d.nrrd               2D labelled atlas for BoundaryFinder
    labels.txt                  ITK-SNAP label description table

Every cross-file invariant the library depends on is maintained simultaneously
and asserted at build time (see ``_check_invariants``), so an invalid override
fails immediately with a readable message rather than as a reshape error deep
inside library code.

The on-disk format was established by inspecting the real assets directly. Two
properties are easy to get wrong and are called out where they are relied on
below: the flat volume lookup is populated *only at streamline start voxels*
(~0.12% fill), and a streamline's layer thicknesses sum to its arc length plus
exactly one voxel.

This module is the format description. Everything it assumes is restated as an
assertion against the real files in ``tests/test_real_data.py``, so a claim
here that stops being true fails a test rather than quietly misleading the next
reader. ``docs/decisions/0001-mini-ccf-fixtures.md`` has the reasoning behind
the approach.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import nrrd
import numpy as np

# ---------------------------------------------------------------------------
# Dimensions
#
# These are constrained rather than chosen. Each is justified here so that
# changing one is a deliberate act rather than a guess.
# ---------------------------------------------------------------------------

#: Volume shape, (x rostro-caudal, y dorso-ventral, z left-right).
#:
#: The three axes are mutually distinct and ordered to mirror the real atlas's
#: proportions (1320 > 1140 > 800, i.e. x > z > y). A cube would silently pass
#: an axis-transposition bug, and two different code paths unravel indices
#: against different shapes, so the asymmetry is load-bearing.
VOLUME_SHAPE = (14, 10, 12)

#: Voxel size in microns. Isotropic, matching the real 10um assets.
RESOLUTION = (10, 10, 10)

#: Rostro-caudal positions of streamlines. Spread rather than contiguous so a
#: nearest-streamline search has a non-degenerate answer.
X_POSITIONS = (3, 5, 7, 9)

#: Lateral positions of streamlines that appear in the 2D view.
#:
#: The lateral axis is even (12), which places the midline on a boundary at
#: z=6, and every streamline sits strictly on one side of it, so reflection
#: maps one half onto the other exactly. Three lateral positions is the minimum
#: that gives a nearest-streamline search a non-degenerate answer.
Z_POSITIONS = (1, 2, 3)

#: Lateral positions of streamlines that exist in the paths file but are *not*
#: referenced by the view. The real assets have 1.48M streamlines of which only
#: ~356K appear in any one view; without this the "nearest streamline within
#: the view" fallback in ``_calculate_2d_coordinates`` is unreachable.
EXTRA_Z_POSITIONS = (4,)

#: First and last y planes are outside cortex.
#:
#: This provides genuine unmappable coordinates, and guarantees no streamline
#: voxel has flat index zero -- critical, because zero is simultaneously the
#: padding value in the streamline array and the cell that maximum and minimum
#: projection overwrite with a sentinel.
Y_FIRST = 1
Y_LAST = 8

#: Padded width of the streamline array. Heavily constrained: see
#: ``_check_padded_length``.
PADDED_LENGTH = 12

#: 2D view size, (lateral, rostro-caudal).
#:
#: The first dimension is even so the opposite-hemisphere offset halves
#: cleanly, and all view content sits within the retained half so that cropping
#: loses nothing and concatenation reproduces the full view size exactly. The
#: two dimensions differ so an axis mixup is caught.
VIEW_SIZE = (8, 5)

#: Rectangular regions of the 2-D labelled atlas, as ``(rows, columns)``.
#:
#: Each is inset from every edge of the view, so ``find_contours`` returns a
#: closed loop whose corners are computable by hand. They differ in size and
#: position so an axis mixup is caught, and they straddle the halfway row so
#: trimming for the other hemisphere retains one and removes the other.
ATLAS_REGIONS = {
    "AAA": (slice(1, 3), slice(1, 4)),  # rows 1-2, columns 1-3
    "BBB": (slice(5, 7), slice(1, 3)),  # rows 5-6, columns 1-2
}

#: Reference layer thicknesses in microns.
#:
#: These are the real values, derived from ``avg_layer_depths.json`` exactly as
#: ``docs/source/guide.rst`` tells users to derive them (successive differences
#: of the cumulative layer tops). Using the real *ratios* is what makes the
#: PADDED_LENGTH constraint below meaningful at fixture scale.
_AVG_LAYER_TOPS = {
    "2/3": 116.8406715462,
    "4": 349.9050202564,
    "5": 477.8605504893,
    "6a": 717.1835081307,
    "6b": 909.8772394508,
    "wm": 957.0592130899,
}

LAYER_KEYS = (
    "Isocortex layer 1",
    "Isocortex layer 2/3",  # contains a slash: a nested group in the HDF5 file
    "Isocortex layer 4",
    "Isocortex layer 5",
    "Isocortex layer 6a",
    "Isocortex layer 6b",
)


def reference_layer_thicknesses():
    """The reference thickness dict, as ``docs/source/guide.rst`` builds it."""
    tops = [0.0] + [_AVG_LAYER_TOPS[k] for k in ("2/3", "4", "5", "6a", "6b", "wm")]
    return {k: tops[i + 1] - tops[i] for i, k in enumerate(LAYER_KEYS)}


# ---------------------------------------------------------------------------
# The built atlas
# ---------------------------------------------------------------------------


@dataclass
class MiniCCF:
    """A built mini-CCF: file paths plus the arrays used to derive expectations.

    Tests assert against values derived from these attributes rather than
    against golden numbers, so an assertion states what the domain requires
    rather than what the code currently returns.
    """

    root: Path

    surface_paths_file: str
    view_lookup_file: str
    layer_thickness_file: str
    closest_surface_voxel_file: str
    atlas_file: str
    labels_file: str

    volume_shape: tuple
    resolution: tuple
    view_size: tuple

    #: (n_paths, padded_length) uint32 array of flat voxel indices, zero-padded.
    paths: np.ndarray
    #: (n_paths, 3) voxel coordinates of each streamline's first (pia) voxel.
    path_starts: np.ndarray
    #: (n_view_rows, 2) int32: column 0 flat view index, column 1 flat volume index.
    view_lookup: np.ndarray
    #: Path index for each row of ``view_lookup``, in view_lookup row order.
    view_path_indices: np.ndarray
    #: Path indices that appear somewhere in ``view_lookup``.
    in_view_path_indices: np.ndarray
    #: Path indices that exist in the paths file but not in the view.
    out_of_view_path_indices: np.ndarray
    #: Reference layer thickness dict, suitable to pass as ``layer_thicknesses``.
    layer_thicknesses: dict
    #: (n_paths, 3) per layer key -- start, end, thickness in microns.
    path_layer_thickness: dict
    #: Path index whose layer 4 is deliberately absent.
    absent_layer_path_index: int
    #: The layer key that is absent from ``absent_layer_path_index``.
    absent_layer_key: str
    #: Acronyms present in the labels file, in index order.
    label_acronyms: tuple
    #: Acronym that is in the labels file but has no pixels in the atlas.
    absent_region_acronym: str
    #: The 2D atlas array written to ``atlas_file``.
    atlas: np.ndarray
    #: ``{acronym: (rows, columns)}`` for the rectangles in ``atlas``.
    atlas_regions: dict

    path_length: int = field(default=0)

    # -- helpers for deriving expectations -------------------------------

    def volume(self, dtype=np.float64):
        """A fresh zero volume of the right shape on *every* call.

        ``project_volume`` no longer writes into its input (issue #20), but a
        volume is still mutable caller state, so call this rather than caching
        the result and letting tests share one.
        """
        return np.zeros(self.volume_shape, dtype=dtype)

    def flat_index(self, x, y, z):
        """Flat index of a voxel, in the same convention as the paths file."""
        return int(np.ravel_multi_index((x, y, z), self.volume_shape))

    def path_voxels(self, path_index):
        """(n, 3) voxel coordinates of one streamline, padding removed."""
        p = self.paths[path_index, :]
        p = p[p > 0]
        return np.array(np.unravel_index(p, self.volume_shape)).T

    def path_microns(self, path_index):
        """(n, 3) micron coordinates of one streamline's voxel corners."""
        return self.path_voxels(path_index) * np.array(self.resolution)

    def path_arc_length(self, path_index):
        """Arc length of one streamline in microns."""
        xyz = self.path_microns(path_index)
        return float(np.sqrt((np.diff(xyz, axis=0) ** 2).sum(axis=1)).sum())

    def view_pixel_for_path(self, path_index):
        """``(row, col)`` of the view pixel a streamline projects to.

        Where a streamline is referenced by more than one view row -- the tied
        case -- this returns the pixel of the *first* such row, which is the
        one a stable sort resolves to. Assertions can therefore be exact
        integer equalities rather than tolerances.
        """
        rows = np.flatnonzero(self.view_path_indices == path_index)
        if len(rows) == 0:
            raise ValueError(f"path {path_index} does not appear in the view")
        flat = int(self.view_lookup[rows[0], 0])
        return tuple(int(v) for v in np.unravel_index(flat, self.view_size))

    def view_pixels_for_path(self, path_index):
        """All ``(row, col)`` view pixels a streamline is referenced by."""
        rows = np.flatnonzero(self.view_path_indices == path_index)
        return [
            tuple(int(v) for v in np.unravel_index(int(self.view_lookup[r, 0]), self.view_size))
            for r in rows
        ]

    def tied_view_keys(self):
        """Flat volume indices referenced by more than one view row."""
        keys, counts = np.unique(self.view_lookup[:, 1], return_counts=True)
        return keys[counts > 1]

    def coord_on_path(self, path_index, step):
        """Micron coordinate at voxel ``step`` along a streamline.

        The coordinate lands on a voxel *corner*, for which the geometric
        offset residual in ``_calculate_2d_coordinates`` is exactly zero, so
        the projected view coordinate is exactly the expected pixel.
        """
        return self.path_microns(path_index)[step, :].astype(float)


# ---------------------------------------------------------------------------
# Build-time invariant checks
# ---------------------------------------------------------------------------


def _check_padded_length(padded_length, layer_thicknesses):
    """The padded length is constrained by layer-normalized slab projection.

    ``_project_volume_normalized_layers`` builds one interpolation block per
    layer by rounding ``padded_length * thickness / total`` , stacks them, and
    reshapes the result into a final dimension of ``padded_length``. The
    rounded blocks must therefore sum to exactly ``padded_length``, and no
    block may be zero or that layer vanishes from the output.

    Swept against the real layer ratios, only 11, 12, 16, 18, 20, 21, 22 and 24
    satisfy both conditions; 12 is the default because it is the smallest that
    gives layer 4 two voxels rather than one.
    """
    total = float(np.sum(list(layer_thicknesses.values())))
    blocks = {
        k: int(np.round(padded_length * t / total)) for k, t in layer_thicknesses.items()
    }
    block_sum = sum(blocks.values())
    if block_sum != padded_length:
        raise ValueError(
            f"padded_length={padded_length} is not usable: the per-layer blocks "
            f"{list(blocks.values())} sum to {block_sum}, not {padded_length}. "
            f"`Isocortex3dProjector._project_volume_normalized_layers` reshapes the "
            f"stacked blocks into a final dimension of {padded_length} and would "
            f"raise a reshape error. Usable values for these layer ratios: "
            f"11, 12, 16, 18, 20, 21, 22, 24."
        )
    empty = [k for k, v in blocks.items() if v == 0]
    if empty:
        raise ValueError(
            f"padded_length={padded_length} gives zero voxels to {empty}, so that "
            f"layer would vanish from a normalized_layers projection. Use 11 or "
            f"more (12 is the default)."
        )
    return blocks


def _check_invariants(mini, flat_lookup, closest):
    """Assert every cross-file invariant the library depends on."""
    shape = mini.volume_shape
    x_size, y_size, z_size = shape

    if len(set(shape)) != 3:
        raise ValueError(
            f"volume_shape {shape} must have three mutually distinct axes; a cube "
            f"silently passes an axis-transposition bug."
        )
    if z_size % 2 != 0:
        raise ValueError(
            f"the lateral axis (volume_shape[2]={z_size}) must be even so the "
            f"midline falls on a voxel boundary and reflection is exact."
        )

    starts = mini.path_starts
    if not np.all(starts[:, 2] < z_size / 2):
        raise ValueError(
            f"every streamline must sit strictly on one side of the midline "
            f"(z < {z_size / 2}); got lateral positions {sorted(set(starts[:, 2].tolist()))}."
        )
    # The nearest-streamline search only ever searches over streamlines the
    # view references, so it is the in-view spread that has to be non-degenerate.
    in_view_lateral = set(starts[mini.in_view_path_indices, 2].tolist())
    if len(in_view_lateral) < 3:
        raise ValueError(
            f"at least three distinct lateral positions are needed for a "
            f"nearest-streamline search to have a non-degenerate answer; the "
            f"view covers {sorted(in_view_lateral)}."
        )

    used = mini.paths[mini.paths > 0]
    if used.size == 0:
        raise ValueError("no streamline voxels were written")
    if 0 in mini.paths[:, 0].tolist():
        raise ValueError(
            "a streamline starts at flat index 0, which is also the padding value "
            "and the cell max/min projection overwrites with a sentinel."
        )

    # Flat volume lookup: populated only at streamline start voxels.
    populated = np.flatnonzero(flat_lookup != -1)
    expected = np.sort(mini.paths[:, 0].astype(np.int64))
    if not np.array_equal(np.sort(populated), expected):
        raise ValueError(
            "the flat volume lookup must be populated at streamline start voxels "
            "and hold the sentinel -1 everywhere else."
        )
    # Each start maps to its own path index; every later voxel on the same
    # streamline still returns the sentinel.
    for i in range(mini.paths.shape[0]):
        row = mini.paths[i, :]
        row = row[row > 0]
        if flat_lookup[int(row[0])] != i:
            raise ValueError(
                f"flat lookup at streamline {i}'s start voxel is "
                f"{flat_lookup[int(row[0])]}, not {i}."
            )
        if len(row) > 1 and np.any(flat_lookup[row[1:].astype(np.int64)] != -1):
            raise ValueError(
                f"streamline {i} has a non-start voxel populated in the flat "
                f"lookup; the real file populates start voxels only."
            )

    # View lookup entries reference genuine streamline starts.
    start_set = set(mini.paths[:, 0].tolist())
    for key in mini.view_lookup[:, 1].tolist():
        if key not in start_set:
            raise ValueError(
                f"view lookup references flat volume index {key}, which is not a "
                f"streamline start voxel."
            )
    if mini.view_size[0] % 2 != 0:
        raise ValueError(
            f"view_size[0]={mini.view_size[0]} must be even so the "
            f"opposite-hemisphere offset halves cleanly."
        )
    if mini.view_size[0] == mini.view_size[1]:
        raise ValueError(
            f"the two view dimensions must differ so an axis mixup is caught; "
            f"got {mini.view_size}."
        )
    rows = np.unravel_index(mini.view_lookup[:, 0], mini.view_size)[0]
    if rows.max() >= mini.view_size[0] // 2:
        raise ValueError(
            f"all view content must sit within the retained half (rows < "
            f"{mini.view_size[0] // 2}) so cropping loses nothing; content reaches "
            f"row {rows.max()}."
        )

    # Tied view keys: the earlier row must carry the smaller view index, which
    # is the contract a stable sort preserves and an unstable one may not.
    order = np.argsort(mini.view_lookup[:, 1], kind="stable")
    s = mini.view_lookup[order]
    boundaries = np.flatnonzero(np.r_[True, np.diff(s[:, 1]) != 0])
    ends = np.r_[boundaries[1:], len(s)]
    n_tied = 0
    for a, b in zip(boundaries, ends):
        if b - a > 1:
            n_tied += 1
            if not np.all(np.diff(s[a:b, 0].astype(np.int64)) > 0):
                raise ValueError(
                    "view lookup rows sharing a volume index must be in increasing "
                    "view-index order, matching the real assets."
                )
    if n_tied == 0:
        raise ValueError(
            "the view lookup has no tied keys, so the stable-sort defect is "
            "unreachable. The real flatmap and rotated views have ties."
        )

    # Atlas regions must be inset from every edge, or their contours are open
    # and there is nothing exact to assert against.
    for acronym, (rows, cols) in mini.atlas_regions.items():
        r0, r1, _ = rows.indices(mini.atlas.shape[0])
        c0, c1, _ = cols.indices(mini.atlas.shape[1])
        if r0 < 1 or r1 > mini.atlas.shape[0] - 1 or c0 < 1 or c1 > mini.atlas.shape[1] - 1:
            raise ValueError(
                f"atlas region {acronym} touches an edge of the {mini.atlas.shape} "
                f"atlas, so its contour would be open rather than a closed loop."
            )
        if not mini.atlas[rows, cols].size:
            raise ValueError(f"atlas region {acronym} is empty.")

    # Closest-surface-voxel lookup.
    if not np.all(np.diff(closest[:, 0].astype(np.int64)) > 0):
        raise ValueError(
            "the closest-surface-voxel lookup's first column must be strictly "
            "increasing; `_matching_voxel_indices` searchsorts it with no sorter."
        )
    for target in np.unique(closest[:, 1]).tolist():
        if target not in start_set:
            raise ValueError(
                f"closest-surface-voxel target {target} is not a streamline start."
            )

    # Layer thicknesses: contiguous, and summing to arc length plus one voxel.
    one_voxel = float(np.mean(mini.resolution))
    for i in range(mini.paths.shape[0]):
        prev_end = 0.0
        total = 0.0
        for k in LAYER_KEYS:
            start, end, thick = mini.path_layer_thickness[k][i, :]
            if start == 0 and end == 0:
                continue  # layer absent from this streamline
            if not np.isclose(start, prev_end, atol=1e-3):
                raise ValueError(
                    f"layer rows for streamline {i} are not contiguous: {k} starts "
                    f"at {start} but the previous layer ended at {prev_end}."
                )
            if not np.isclose(end - start, thick, atol=1e-3):
                raise ValueError(
                    f"layer {k} of streamline {i} has end-start={end - start} but "
                    f"thickness={thick}."
                )
            prev_end = end
            total += float(thick)
        expected_total = mini.path_arc_length(i) + one_voxel
        if not np.isclose(total, expected_total, atol=1e-2):
            raise ValueError(
                f"streamline {i}'s layer thicknesses sum to {total}um but its arc "
                f"length plus one voxel is {expected_total}um. Measured on the real "
                f"assets, the two are equal to within floating point: layer totals "
                f"add half a voxel of pia and half of white matter. A fixture that "
                f"ignores this pushes every depth query past the final layer "
                f"boundary, so layer-normalized depths all return NaN."
            )


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def build_mini_ccf(
    root,
    volume_shape=VOLUME_SHAPE,
    resolution=RESOLUTION,
    x_positions=X_POSITIONS,
    z_positions=Z_POSITIONS,
    extra_z_positions=EXTRA_Z_POSITIONS,
    y_first=Y_FIRST,
    y_last=Y_LAST,
    padded_length=PADDED_LENGTH,
    view_size=VIEW_SIZE,
    n_tied_view_rows=2,
    layer_thicknesses=None,
    absent_layer_path_index=5,
    absent_layer_key="Isocortex layer 4",
    atlas_regions=None,
):
    """Build a coherent mini-CCF into ``root`` and return a :class:`MiniCCF`.

    Every parameter perturbs exactly one structural property, so a test can
    vary one dimension without rebuilding the rest by hand. Invalid
    combinations raise ``ValueError`` here rather than failing deep inside
    library code.

    Parameters
    ----------
    root : path-like
        Directory to write the six files into. Usually pytest's ``tmp_path``.
    volume_shape : 3-tuple, default (14, 10, 12)
        Shape of the miniature volume, (x, y, z).
    resolution : 3-tuple, default (10, 10, 10)
        Voxel size in microns.
    x_positions, z_positions : tuples
        Rostro-caudal and lateral positions of the streamlines that appear in
        the 2D view. Their product is the number of in-view streamlines.
    extra_z_positions : tuple
        Lateral positions of streamlines present in the paths file but absent
        from the view, which is what makes the nearest-streamline fallback
        reachable.
    y_first, y_last : int
        First and last dorso-ventral plane occupied by a streamline. Planes
        outside this range are outside cortex and yield unmappable coordinates.
    padded_length : int, default 12
        Padded width of the streamline array. Constrained; see
        ``_check_padded_length``.
    view_size : 2-tuple, default (8, 5)
        Shape of the 2D view.
    n_tied_view_rows : int, default 2
        How many extra view rows reference an already-referenced surface voxel.
        Ties are what make the stable-sort defect reachable.
    layer_thicknesses : dict, optional
        Reference layer thicknesses in microns. Defaults to the real values.
    absent_layer_path_index : int or None, default 5
        Streamline for which one layer is deliberately absent, so the
        documented behaviour of leaving a gap can be verified rather than
        assumed. Pass None to give every streamline every layer.
    absent_layer_key : str
        Which layer is absent from that streamline.
    atlas_regions : dict, optional
        ``{acronym: (rows, columns)}`` rectangles for the 2-D labelled atlas.
        Defaults to :data:`ATLAS_REGIONS`.

    Returns
    -------
    mini : MiniCCF
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if layer_thicknesses is None:
        layer_thicknesses = reference_layer_thicknesses()
    if atlas_regions is None:
        atlas_regions = ATLAS_REGIONS

    _check_padded_length(padded_length, layer_thicknesses)

    path_length = y_last - y_first + 1
    if path_length > padded_length:
        raise ValueError(
            f"streamlines span {path_length} voxels (y {y_first}..{y_last}) but the "
            f"padded length is only {padded_length}."
        )
    if y_first < 1 or y_last > volume_shape[1] - 2:
        raise ValueError(
            f"streamlines must leave a plane outside cortex at each end of the "
            f"dorso-ventral axis: y must lie within 1..{volume_shape[1] - 2}, got "
            f"{y_first}..{y_last}."
        )

    # -- streamlines --------------------------------------------------------
    # In-view streamlines first, then the out-of-view ones, so path indices are
    # stable when extra_z_positions changes.
    lateral_groups = [(z, True) for z in z_positions] + [
        (z, False) for z in extra_z_positions
    ]
    starts = []
    in_view_flags = []
    for z, in_view in lateral_groups:
        for x in x_positions:
            starts.append((x, y_first, z))
            in_view_flags.append(in_view)
    path_starts = np.array(starts, dtype=int)
    in_view_flags = np.array(in_view_flags, dtype=bool)
    n_paths = len(starts)

    paths = np.zeros((n_paths, padded_length), dtype=np.uint32)
    for i, (x, _, z) in enumerate(path_starts):
        ys = np.arange(y_first, y_last + 1)
        flat = np.ravel_multi_index(
            (np.full_like(ys, x), ys, np.full_like(ys, z)), volume_shape
        )
        paths[i, : len(flat)] = flat

    # -- flat volume lookup: populated only at streamline start voxels ------
    flat_lookup = np.full(int(np.prod(volume_shape)), -1, dtype=np.int32)
    flat_lookup[paths[:, 0].astype(np.int64)] = np.arange(n_paths, dtype=np.int32)

    # -- view lookup --------------------------------------------------------
    # Row = lateral index, column = rostro-caudal index. Generated in
    # (lateral, rostro-caudal) order, which leaves the volume-index column
    # unsorted exactly as the real files are.
    view_rows = []
    view_path_index = []
    in_view_paths = np.flatnonzero(in_view_flags)
    for zi, z in enumerate(z_positions):
        for xi, x in enumerate(x_positions):
            path_index = int(
                np.flatnonzero(
                    (path_starts[:, 0] == x) & (path_starts[:, 2] == z)
                )[0]
            )
            view_flat = int(np.ravel_multi_index((zi, xi), view_size))
            view_rows.append((view_flat, int(paths[path_index, 0])))
            view_path_index.append(path_index)

    # Extra rows that reference an already-referenced surface voxel. Placed in
    # the last retained row of the view with a larger view index than the row
    # they duplicate, matching the real assets, where a stable sort leaves the
    # smallest view index first within every tied group.
    tie_row = view_size[0] // 2 - 1
    for t in range(n_tied_view_rows):
        duplicated = t * (len(x_positions) + 1) % len(view_path_index)
        view_flat = int(np.ravel_multi_index((tie_row, view_size[1] - 1 - t), view_size))
        if view_flat <= view_rows[duplicated][0]:
            raise ValueError(
                "a tie-creating view row must carry a larger view index than the "
                "row it duplicates; enlarge view_size or reduce n_tied_view_rows."
            )
        view_rows.append((view_flat, view_rows[duplicated][1]))
        view_path_index.append(view_path_index[duplicated])

    view_lookup = np.array(view_rows, dtype=np.int32)
    view_path_index = np.array(view_path_index, dtype=int)

    # -- closest surface voxel lookup --------------------------------------
    # Every voxel on a streamline maps to that streamline's start. Voxels on
    # the planes outside cortex get no entry at all, so they resolve to the
    # missing-value sentinel and are genuinely unmappable.
    closest_rows = []
    for i in range(n_paths):
        for flat in paths[i, :][paths[i, :] > 0].tolist():
            closest_rows.append((flat, int(paths[i, 0])))
    closest = np.array(sorted(closest_rows), dtype=np.uint32)

    # -- per-streamline layer thicknesses ----------------------------------
    # Scaled so each streamline's layers sum to its own arc length plus one
    # voxel, which is the relation the real files hold to within floating
    # point.
    one_voxel = float(np.mean(resolution))
    ref_total = float(np.sum(list(layer_thicknesses.values())))
    path_layer_thickness = {k: np.zeros((n_paths, 3), dtype=np.float32) for k in LAYER_KEYS}
    for i in range(n_paths):
        xyz = np.array(np.unravel_index(paths[i, :][paths[i, :] > 0], volume_shape)).T * np.array(
            resolution
        )
        arc = float(np.sqrt((np.diff(xyz, axis=0) ** 2).sum(axis=1)).sum())
        target_total = arc + one_voxel

        keys = list(LAYER_KEYS)
        if absent_layer_path_index is not None and i == absent_layer_path_index:
            keys = [k for k in keys if k != absent_layer_key]
        # Redistribute proportionally over the layers this streamline has, so
        # the total still matches arc length plus a voxel and the layers stay
        # contiguous.
        present_total = float(sum(layer_thicknesses[k] for k in keys))
        cursor = 0.0
        for k in keys:
            thick = layer_thicknesses[k] / present_total * target_total
            path_layer_thickness[k][i, :] = (cursor, cursor + thick, thick)
            cursor += thick
    del ref_total

    # -- 2D atlas and label table ------------------------------------------
    # Two rectangles, each inset from every edge so `find_contours` returns a
    # closed loop whose corners are computable by hand: a region occupying
    # rows r0..r1 and columns c0..c1 has its contour at r0-0.5 .. r1+0.5 and
    # c0-0.5 .. c1+0.5. A region touching the array edge would give an open
    # contour instead, which is a much weaker thing to assert against.
    #
    # The two differ in both size and position so an axis mixup is caught, and
    # they sit either side of the halfway row, so trimming for the other
    # hemisphere retains one and removes the other entirely.
    atlas = np.zeros(view_size, dtype=np.uint32)
    atlas[atlas_regions["AAA"]] = 1
    atlas[atlas_regions["BBB"]] = 2
    label_acronyms = ("AAA", "BBB", "CCC")
    absent_region_acronym = "CCC"

    # -- write the files ----------------------------------------------------
    surface_paths_file = root / "surface_paths_10_v3.h5"
    with h5py.File(surface_paths_file, "w") as f:
        # The current-generation file carries no file-level attributes and no
        # 3D "volume lookup" dataset; only the flattened form, whose single
        # attribute records the original volume shape.
        f.create_dataset("paths", data=paths)
        dset = f.create_dataset("volume lookup flat", data=flat_lookup)
        dset.attrs["original shape"] = np.array(volume_shape, dtype=np.int64)

    view_lookup_file = root / "view_lookup.h5"
    with h5py.File(view_lookup_file, "w") as f:
        f.create_dataset("view lookup", data=view_lookup)
        # Origin and spacing are byte strings while size and view size are
        # integers. The loading code decodes spacing, so a fixture that wrote
        # it as integers would pass while the real file crashed.
        f.attrs["origin"] = np.array([b"0", b"0", b"0"])
        f.attrs["size"] = np.array(volume_shape, dtype=np.int64)
        f.attrs["spacing"] = np.array([str(r).encode() for r in resolution])
        f.attrs["view size"] = np.array(view_size, dtype=np.int64)

    layer_thickness_file = root / "cortical_layers_10_v2.h5"
    with h5py.File(layer_thickness_file, "w") as f:
        for k in LAYER_KEYS:
            # "Isocortex layer 2/3" contains a slash, so this creation makes a
            # nested group -- which is what the real file has.
            f.create_dataset(k, data=path_layer_thickness[k])

    closest_surface_voxel_file = root / "closest_surface_voxel.h5"
    with h5py.File(closest_surface_voxel_file, "w") as f:
        f.create_dataset("closest surface voxel", data=closest)

    atlas_file = root / "atlas_2d.nrrd"
    nrrd.write(str(atlas_file), atlas)

    labels_file = root / "labels.txt"
    # Deliberately ragged whitespace, as the real ITK-SNAP table is aligned
    # with a mixture of spaces and tabs.
    labels_file.write_text(
        "1   255   0   0  1  1  1\tAAA\n"
        "2     0 255   0  1  1  1   BBB\n"
        "3     0   0 255  1  1  1  CCC\n"
    )

    mini = MiniCCF(
        root=root,
        surface_paths_file=str(surface_paths_file),
        view_lookup_file=str(view_lookup_file),
        layer_thickness_file=str(layer_thickness_file),
        closest_surface_voxel_file=str(closest_surface_voxel_file),
        atlas_file=str(atlas_file),
        labels_file=str(labels_file),
        volume_shape=tuple(volume_shape),
        resolution=tuple(resolution),
        view_size=tuple(view_size),
        paths=paths,
        path_starts=path_starts,
        view_lookup=view_lookup,
        view_path_indices=view_path_index,
        in_view_path_indices=in_view_paths,
        out_of_view_path_indices=np.flatnonzero(~in_view_flags),
        layer_thicknesses=layer_thicknesses,
        path_layer_thickness=path_layer_thickness,
        absent_layer_path_index=absent_layer_path_index,
        absent_layer_key=absent_layer_key,
        label_acronyms=label_acronyms,
        absent_region_acronym=absent_region_acronym,
        atlas=atlas,
        atlas_regions=dict(atlas_regions),
        path_length=path_length,
    )

    _check_invariants(mini, flat_lookup, closest)
    return mini


def write_avg_layer_depths(root):
    """Write an ``avg_layer_depths.json`` alongside the mini-CCF."""
    path = Path(root) / "avg_layer_depths.json"
    path.write_text(json.dumps(_AVG_LAYER_TOPS))
    return str(path)
