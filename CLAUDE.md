# CLAUDE.md

<!-- codebase-summarizer: begin -->

`ccf-streamlines` is a pure-Python library (no CLI, no service) for projecting Allen
Mouse Common Coordinate Framework (CCFv3) volumes and 3D coordinates onto 2D flatmaps
and flattened 3D "slabs", using precomputed *streamlines* that run from the pia to the
white matter of isocortex. It is imported by Allen Institute analysis code and notebooks;
everything it does is an array transform against large HDF5/NRRD reference files that the
user downloads separately (`docs/source/data_files.rst` has the download links). No
network access, no database, no state outside the caller's arrays.

## Commands

    uv sync                       # install; requires-python >=3.9
    uv run pytest                 # whole suite: 7 tests, ~3s (pyproject sets testpaths=tests, pythonpath=src)
    cd docs && make html          # Sphinx docs -> docs/build (needs docs/requirements.txt: sphinx 5.2.3)

There is no lint, format, or typecheck configuration in the repo, and no CI test job —
`.github/workflows/python-publish.yml` is the only workflow and it only builds and
publishes to PyPI on GitHub release. Merges are gated by nothing automated.

## Layout

- `src/ccf_streamlines/` — the entire package; src-layout, built by `uv_build` (`pyproject.toml`).
- `tests/` — pytest. Covers `coordinates.py` and `BoundaryFinder`; the projector
  classes are untested because they need the multi-hundred-MB reference files.
- `docs/source/guide.rst` — the real user documentation: worked examples for each
  projector class, with expected output images. Read this before changing a public
  signature; it is the closest thing to an integration test.
- `data_file_info.md` (repo root, not in the Sphinx build) — internal notes on the
  reference data files, including the on-`/allen` paths and the shapes of each NRRD view.

## Key abstractions

- **Streamline paths** — an `(n_paths, max_len)` int array read from the `"paths"` dataset
  of `surface_paths_*.h5`. Each row holds *flattened* voxel indices into a
  `(1320, 800, 1140)` volume, zero-padded on the right. Index `0` is reserved as "no
  voxel", so `paths > 0` is the validity mask used everywhere.
- **`view lookup`** — an `(n, 2)` array in the projection HDF5: column 0 is a flat index
  into the 2D view, column 1 is a flat index into the 3D volume. Projection is just
  `view[lookup[:,0]] = f(volume.flat[paths])`.
- `Isocortex2dProjector` (`projection.py`) — flattens a volume to 2D by reducing each
  streamline (`max`/`min`/`mean`/`sum`). Base class for the others.
- `Isocortex3dProjector` (`projection.py`) — subclass producing a 3D slab, with three
  `thickness_type` modes (`unnormalized`, `normalized_full`, `normalized_layers`); the
  normalized modes work by building one giant `np.interp` over all streamlines at once
  (`_project_volume_normalized_full`, `_project_volume_normalized_layers`).
- `IsocortexCoordinateProjector` (`projection.py`) — the inverse direction: takes
  `(N, 3)` micron coordinates and returns flattened coordinates. Unlike the volume
  projectors it loops per-coordinate in Python, so it is by far the slowest path.
- `LineString3D` (`linestring3d.py`) — a minimal 3D analogue of shapely's `LineString`
  (`project`, `offset_of_point`, `rotation_to_vector`). Used only by
  `IsocortexCoordinateProjector` to get sub-voxel depth and x-y offset.
- `BoundaryFinder` (`projection.py`) — unrelated to streamlines; reads a projected atlas
  NRRD plus an ITK-SNAP label text file and returns region outlines via
  `skimage.measure.find_contours`, for overlaying on a projection.

## How a 2D projection runs

1. `Isocortex2dProjector.__init__` reads `view lookup`, `view size`, `spacing` from the
   projection file, then `_load_and_sort_paths` reorders `paths` to match the view. That
   reorder reads `volume lookup flat` in 1000-element chunks (`projection.py:102`) —
   deliberately, to keep memory down (commit ef0cd04); the same chunking is repeated in
   `IsocortexCoordinateProjector._path_lookup_chunked`.
2. `project_volume(volume, kind)` dispatches on `self.hemisphere`. For `"right"` and
   `"both"` it re-projects `np.flip(volume, axis=2)` and flips the result back.
3. `_project_volume_to_view` does the actual reduction: `volume.flat[self.paths].max(axis=1)`
   scattered into `projected_volume.flat[self.view_lookup[:, 0]]`.
4. `view_space_for_other_hemisphere` trims `n` columns off the right edge before the two
   hemispheres are concatenated; `HEMISPHERE_SPACE_VIEW_LOOKUP` (`projection.py:14`) holds
   the per-view presets (e.g. `flatmap_butterfly: 184`).

## Conventions

- NumPy-style docstrings with `Parameters`/`Returns` on every public function and class;
  AST-checked all 9 modules, the only two without one are `BoundaryFinder.region_masks`
  (`projection.py:628`) and `angle.vector_to_3d_affine_matrix` (`angle.py:8`).
- Argument validation is explicit `if x not in {...}: raise ValueError(f"...")` at the top
  of public methods; there is no custom exception type anywhere, and the only `except` in
  `src/` is the `PackageNotFoundError` guard in `__init__.py`. Follow the `ValueError`
  pattern and let everything else propagate.
- No type annotations anywhere in `src/`. Do not add them piecemeal.
- Modules are flat function/class collections; `__init__.py` exports nothing but
  `__version__` (via `importlib.metadata`), so callers always import the submodule.
- Diagnostics go through the root logger as `logging.info(...)` / `logging.warning(...)`,
  not a module-level `logger` — but progress bars print directly (`print("loading path
  information")` in `projection.py:105` and `:976`, plus `tqdm`), so the library is not
  silent by default.
- Volume shape `(1320, 800, 1140)` and resolution `(10, 10, 10)` are repeated as literal
  defaults in `angle.py`, `dataset.py`, and `morphology.py`; there is no shared constant.
- The version lives only in `pyproject.toml`. `docs/source/conf.py` regex-scrapes it and
  `__init__.py` reads it from installed metadata. `.bumpversion.cfg` was deleted in the
  uv conversion (a9f23cb), so `bump2version` no longer works — edit `pyproject.toml` by hand.

## Gotchas

- **`project_volume` mutates the caller's array.** For `kind="max"`/`"min"` it writes a
  sentinel into `volume.flat[0]` (`projection.py:179`, `:1219`) and never restores it. With
  `hemisphere="both"` the second pass writes through the `np.flip` *view*, so
  `volume[0, 0, -1]` is clobbered too (verified). Copy the volume before projecting if you
  need it intact.
- **Unknown `kind` fails silently or obscurely.** `_project_volume_to_view` has no `else`
  branch, so a typo'd `kind` returns an all-zeros view; `IsocortexEntireProjector.project_volume`
  hits `UnboundLocalError` on `values` instead.
- `region_masks` accepts `hemisphere="left_for_both"` but `region_boundaries` does not
  handle it (it falls through to the un-shifted "left" case). The two methods share
  `_validate_inputs` but not their supported values.
- `_matching_voxel_indices` (`projection.py:1264`) uses `np.searchsorted` with no `sorter`
  in most call sites, so the reference file's lookup column is assumed pre-sorted. Nothing
  checks this; an unsorted reference file yields wrong voxels, not an error.
- `metrics.measure_streamline_layer_thicknesses` re-implements path deduplication inline
  with a Python loop instead of calling `processing.remove_duplicate_voxels_from_paths`,
  which is vectorized and has zero callers in the repo. The layer-sorting hack right below
  it is flagged as such in its own comment (`metrics.py:68`).
- `.python-version` (pinning 3.9) is **gitignored**, so a fresh clone is not pinned — `uv sync`
  there resolves to whatever interpreter uv finds (3.13 in this checkout). Nothing tests 3.9.
- Dead imports: `itertools` in `projection.py`; `h5py`, `nrrd`, `pandas`, `logging`, and
  `tqdm` in `metrics.py` (only `numpy` is used there).
- `docs/source/reference/processing.rst` titles itself `ccf_streamlines.projection` — wrong
  module name in the heading.
- A stale, untracked `ccf_streamlines.egg-info/` sits in the repo root from the pre-uv
  setuptools build. It is not the build source; `uv_build` is (`pyproject.toml`).

<!-- codebase-summarizer: end; generated 2026-08-28, commit c91140a, 39 claims -->
