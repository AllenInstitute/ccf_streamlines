# Streamline-related file formats

Internal notes on the on-disk format of the reference assets. This document
describes what the **current-generation** files actually contain, verified by
inspecting them rather than by reading the previous version of this note, which
had drifted: it described a superseded generation of the streamline file, named
a dataset that no longer exists, and omitted the attribute the current code
depends on to recover volume shape.

For the *user-facing* list of files and download links, see
`docs/source/data_files.rst`. That page is correct and already names the
current-generation files. This document is about their internal structure.

## Where the files live

    /allen/programs/celltypes/workgroups/ivscc/nathang/flatmap/handoff   (current)
    /allen/aibs/ccf/2017_integration/handoff                            (older)

**These two trees are no longer equivalent.** The first is the superset and
holds the current-generation assets. The older shared tree lacks
`surface_paths_10_v3.h5`, `cortical_layers_10_v2.h5`, and
`closest_surface_voxel_lookup.h5`. Use the first, or download from
`download.alleninstitute.org` as `docs/source/data_files.rst` describes.

The CCF volume at 10 micron voxels is 1320 x 800 x 1140:

* 1320 is x - rostral to caudal (anterior-posterior)
* 800 is y - superior to inferior (dorso-ventral)
* 1140 is z - left to right (lateral)

Only the left hemisphere is populated in the streamline and metric files.
Code reflects right-hemisphere coordinates across the midline at z = 570
before looking anything up.

## Which generation of each file the code reads

The package reads the current generation only. Older files with the same
purpose are still on disk and are **not** interchangeable.

| Purpose | Current file | Superseded |
|---|---|---|
| Streamline paths + voxel lookup | `surface_paths_10_v3.h5` | `surface_paths_10.h5`, `surface_paths_10_v2.h5` |
| Per-streamline layer depths | `cortical_layers_10_v2.h5` | `cortical_layers_10.h5` |
| Nearest surface voxel | `closest_surface_voxel_lookup.h5` | `closest_surface_voxel_index.nrrd` |

The difference that matters: the superseded surface-paths files store a 3-D
`volume lookup` dataset of the full CCF shape, while v3 stores only a
*flattened* `volume lookup flat` and records the original shape in an
attribute. Code that reads `volume lookup` therefore fails against the current
assets - see the note on `project_path_ordered_data` below.

## streamlines/surface_paths_10_v3.h5

About 452 MiB. **Carries no file-level attributes.** (The superseded
`surface_paths_10.h5` carries `origin`, `size`, and `spacing`; v2 and v3 do
not.)

- dataset `paths`, shape (1476024, 200), dtype uint32

  One row per streamline. Each row holds the *flattened* voxel indices of the
  voxels along that streamline, ordered from pia to white matter, right-padded
  with zeros. Flat index 0 is therefore reserved as "no voxel", and `paths > 0`
  is the validity mask used throughout the package.

- dataset `volume lookup flat`, shape (1203840000,), dtype int32

  A flattened voxel-to-streamline lookup. Cell *i* holds the index into `paths`
  of the streamline that **starts** at flat voxel *i*, or **-1** if no
  streamline starts there.

  - attribute `original shape` = [1320, 800, 1140]

    The only attribute on the file. This is how the package recovers the volume
    shape, since the file-level attributes are gone. Both
    `Isocortex2dProjector._load_and_sort_paths` and
    `IsocortexCoordinateProjector.__init__` read it.

  **The lookup is populated only at streamline start voxels** - roughly 0.12%
  of cells; every other cell holds the sentinel. It is *not* populated at every
  voxel along a streamline. Verified directly: for streamline *i*, the lookup
  at its first voxel returns *i*, while every subsequent voxel on the same
  streamline returns -1.

  This is sufficient because every call site queries only surface voxels,
  having first resolved an arbitrary voxel through the closest-surface-voxel
  lookup below. It is worth knowing when building test fixtures or new tooling:
  filling the lookup along whole streamlines is not what the real file does.

- There is **no** 3-D `volume lookup` dataset in v3.

  `Isocortex2dProjector.project_path_ordered_data` still reads `f["volume lookup"]`
  and therefore raises `KeyError` against the current assets.

## streamlines/closest_surface_voxel_lookup.h5

About 185 MiB.

- dataset `closest surface voxel`, shape (61911881, 2), dtype uint32

  Column 0 is the flattened index of a voxel inside isocortex; column 1 is the
  flattened index of the surface voxel whose streamline best matches it. Many
  voxels share a target - up to a few hundred.

  **Column 0 is sorted and strictly increasing.** `_matching_voxel_indices`
  searches it with `np.searchsorted` and passes no `sorter`, so the ordering is
  assumed, not checked. A voxel absent from column 0 is outside isocortex and
  resolves to the missing-value sentinel 0.

  Note this is a two-column lookup table, not a volume. The superseded
  `closest_surface_voxel_index.nrrd` was a CCF-shaped volume.

## cortical_metrics/cortical_layers_10_v2.h5

About 101 MiB. No file-level attributes.

Six datasets, each of shape (1476024, 3) and dtype float32, one row per
streamline in the same order as `paths`:

    Isocortex layer 1
    Isocortex layer 2/3
    Isocortex layer 4
    Isocortex layer 5
    Isocortex layer 6a
    Isocortex layer 6b

The three columns are **start depth, end depth, and thickness**, in microns
along the streamline from pia. Rows are contiguous: each layer's start equals
the previous layer's end. A layer absent from a given streamline has all three
values set to 0, which is how the code detects absence.

**The key `Isocortex layer 2/3` contains a slash, so it is a genuine nested
group** (`Isocortex layer 2` -> dataset `3`), not a flat name containing a
slash. Creating it with the slashed key produces this structure naturally, and
reading it with the slashed key works, so the nesting is usually invisible.

**Total layer thickness equals the streamline's arc length plus one voxel.**
Measured across the real file, `sum(thicknesses) - arc_length` is 10.000 um
exactly - half a voxel of pia at the top plus half a voxel of white matter at
the bottom. (As a ratio this is about 1.010 for a typical streamline.)

This matters for anyone building fixtures or synthetic data: if layer
thicknesses are not scaled to the streamline's own arc length, every
layer-normalized depth query falls past the final layer boundary and returns
the missing-value sentinel.

## view_lookup/*.h5

One file per view: `back`, `bottom`, `flatmap_butterfly`, `flatmap_dorsal`,
`front`, `medial`, `rotated`, `side`, `top`. A few MiB each.

- dataset `view lookup`, shape (n, 2), dtype int32

  Column 0 is the flattened index into the 2-D view; column 1 is the flattened
  index into the 3-D CCF volume of the surface voxel seen at that pixel. The
  second column is what connects a 2-D location to the streamline start beneath
  it.

  **Column 1 is not sorted**, which is why `_calculate_2d_coordinates` builds
  and passes a `sorter`.

- file attributes, with a type asymmetry the loading code depends on:

  | attribute | type | example |
  |---|---|---|
  | `origin` | array of **byte strings** | `[b'0', b'0', b'0']` |
  | `spacing` | array of **byte strings** | `[b'10', b'10', b'10']` |
  | `size` | array of **integers** | `[1320, 800, 1140]` |
  | `view size` | array of **integers** | `[1140, 1320]` |

  `Isocortex2dProjector.__init__` does `int(d.decode())` over `spacing` and
  indexes `view size` directly. A file writing `spacing` as integers would
  crash there, so fixtures must reproduce the asymmetry.

### Tied keys, and where they occur

Some views map several 2-D pixels onto the same surface voxel, so column 1 has
repeated values. **This is not uniform across views:**

| view | rows | unique col 1 | tied keys | max pixels per voxel |
|---|---|---|---|---|
| back | 85965 | 85965 | 0 | 1 |
| bottom | 85445 | 85445 | 0 | 1 |
| front | 208956 | 208956 | 0 | 1 |
| medial | 165311 | 165311 | 0 | 1 |
| side | 343575 | 343575 | 0 | 1 |
| top | 356866 | 356866 | 0 | 1 |
| flatmap_dorsal | 1014760 | 669683 | 281997 | 10 |
| flatmap_butterfly | 1015244 | 664288 | 283857 | 10 |
| rotated | 471302 | 334745 | 105965 | 11 |

Ties matter because `_matching_voxel_indices` resolves a key by
`np.searchsorted`'s left insertion point, so for a tied key it returns whichever
tied row the sort placed first - and numpy's default quicksort is unstable.
Within every tied group the rows are in **increasing view-index order**
(verified: 283857 tied groups in `flatmap_butterfly`, zero exceptions), so a
stable sort always resolves to the smallest tied view index.

This is the whole of issue #12: results differ by CPU architecture, but only
for the three views that have ties.

## master_updated/labelDescription_ITKSNAPColor.txt

Whitespace-delimited, one row per structure, aligned with a mixture of spaces
and tabs:

    index  R  G  B  x0  x1  x2  acronym

`BoundaryFinder` parses it with `sep=r"\s+"`. The index column matches the
values in the 2-D atlas NRRDs in the same directory.

## Basic operation: a 2-D max projection

1. Read `view lookup` from the view file (e.g. `top.h5`).
2. For each row, take column 1 - a flat volume index of a surface voxel - and
   look it up in `volume lookup flat` to get that voxel's streamline index.
3. Gather those rows from `paths`, in view-lookup row order.
4. For each streamline, take the maximum over the input volume at its voxels
   and write it to the flat view index in column 0.

Step 2 is why the flat lookup only needs entries at streamline starts: column 1
of a view lookup is always a streamline start.
