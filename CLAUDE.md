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
    uv run pytest                 # whole suite; seconds, no data files needed
    CCF_STREAMLINES_TEST_DATA=... uv run pytest -m real_data   # opt-in tier, real assets
    cd docs && make html          # Sphinx docs -> docs/build (needs docs/requirements.txt: sphinx 5.2.3)
    uv run ruff check --fix .     # lint
    uv run ruff format .          # format

Ruff is the only lint/format tool; its configuration is the `[tool.ruff]` block in
`pyproject.toml` (`select = E, W, F, I, UP, B, C4`, `target-version = "py39"`, `E501`
left to the formatter). There is no typecheck configuration, and `src/` carries no type
annotations, so do not add a typechecker piecemeal.
`.github/workflows/tests.yml` gates pull requests across x86-64 and ARM64 on Python
3.9 and 3.13; the architecture dimension is load-bearing, since the tied-view-lookup
defect exists only as a divergence between architectures. Coverage is reported, not
gated. Its second job, `lint`, runs `ruff check` and `ruff format --check` once on
x86-64 — findings are a property of the source, not the machine — installing only the
dev group so the scientific stack is skipped. `python-publish.yml` builds and publishes
to PyPI on GitHub release.

## Layout

- `src/ccf_streamlines/` — the entire package; src-layout, built by `uv_build` (`pyproject.toml`).
- `tests/` — pytest, no data files needed. Tests marked `xfail` are pinned to an open
  bug or an open PR and assert the *correct* behaviour, so `xfail_strict` reports an
  unexpected pass when one is fixed — that is the signal to delete the marker.
- `docs/source/guide.rst` — the real user documentation: worked examples for each
  projector class, with expected output images. Read this before changing a public
  signature; it is the closest thing to an integration test.
- `tests/mini_ccf.py` — builds a ~20 KB structurally faithful atlas into a temp
  directory, so the projector classes are constructible without the multi-hundred-MB
  assets. Also the format description: it is where to look for what the real HDF5
  files contain, and every claim in it is asserted against them by
  `tests/test_real_data.py`.
- `docs/decisions/` — decision records. `0001` explains the fixture strategy.

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
   reorder reads `volume lookup flat` in chunks — deliberately, to keep memory down
   (commit ef0cd04). The size is the `chunk_size` constructor argument, default 1000;
   the same chunking is repeated in `IsocortexCoordinateProjector._path_lookup_chunked`.
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
  (`projection.py:660`) and `angle.vector_to_3d_affine_matrix` (`angle.py:8`).
- Argument validation is explicit `if x not in {...}: raise ValueError(f"...")` at the top
  of public methods; there is no custom exception type anywhere, and the only `except` in
  `src/` is the `PackageNotFoundError` guard in `__init__.py`. Follow the `ValueError`
  pattern and let everything else propagate.
- No type annotations anywhere in `src/`. Do not add them piecemeal.
- Modules are flat function/class collections; `__init__.py` exports nothing but
  `__version__` (via `importlib.metadata`), so callers always import the submodule.
- Diagnostics go through the root logger as `logging.info(...)` / `logging.warning(...)`,
  not a module-level `logger` — but progress bars print directly (`print("loading path
  information")` in `projection.py:126` and `:1022`, plus `tqdm`), so the library is not
  silent by default.
- Volume shape `(1320, 800, 1140)` and resolution `(10, 10, 10)` are repeated as literal
  defaults in `angle.py`, `dataset.py`, and `morphology.py`; there is no shared constant.
- The version lives only in `pyproject.toml`. `docs/source/conf.py` regex-scrapes it and
  `__init__.py` reads it from installed metadata. `.bumpversion.cfg` was deleted in the
  uv conversion (a9f23cb), so `bump2version` no longer works — edit `pyproject.toml` by hand.

## Gotchas

- `project_volume` used to write a `kind="max"`/`"min"` sentinel into the caller's
  `volume.flat[0]` and never restore it (issue #20). The substitution is now made in the
  gathered copy, so the caller's array is left alone — `tests/test_project_volume_mutation.py`
  holds that line.
- `kind` and `scale` are validated at the top of `project_volume` (both projectors) and
  `IsocortexEntireProjector.top_of_streamline_coords` (issue #21). The dispatch chains
  below them still have no `else`, so any *new* accepted value must be added in both
  places or it silently returns zeros / `None`.
- `region_masks` accepts `hemisphere="left_for_both"` but `region_boundaries` does not
  handle it (it falls through to the un-shifted "left" case). The two methods share
  `_validate_inputs` but not their supported values.
- `_matching_voxel_indices` (`projection.py`) resolves voxels with `np.searchsorted`, and
  the *same* binary search decides whether a query matched at all, by comparing against
  the key at the insertion point — do not reintroduce an `np.isin` membership test. It
  walks the whole key column, 61.9M rows in `closest_surface_voxel_lookup.h5`, and
  `angle.find_closest_streamline` calls the helper once per coordinate, so that cost
  would be paid per point (12.5 s each, measured).
- `searchsorted` needs that column ordered and does not verify it, so
  `_check_lookup_is_ordered` does, raising `ValueError` naming the first out-of-order
  entry. It runs inside `_matching_voxel_indices`, covering both the plain column and the
  `sorter` order. The scan is O(rows) — ~74 ms on the 61.9M-row table against 0.03 ms for
  a lookup — so the result is memoised per array in the module-level
  `_checked_lookup_orders` (a `WeakValueDictionary`, keyed on `id`, values weak so an
  entry dies with its array). Dropping that cache silently restores a per-coordinate
  O(rows) cost. The trade-off it buys: a lookup reordered *in place* after its first use
  is not re-checked; `tests/test_matching_voxel_indices.py` pins that.
- The isocortex layer names have one definition, module-level `projection.ISOCORTEX_LAYER_KEYS`
  (issue #26). Both projector classes bind their `ISOCORTEX_LAYER_KEYS` class attribute to it
  and `metrics.LAYER_LABELS` zips it against `metrics.ISOCORTEX_LAYER_STRUCTURE_SET_IDS`, so
  a thickness file `metrics` writes is always keyed the way the projectors read it. Edit the
  one list; `tests/test_layer_key_constants.py` asserts identity, not just equality.
- `IsocortexCoordinateProjector` mirrors a right-hemisphere query by reflecting the micron
  coordinate (`z_size * resolution - u`) and then taking the voxel that mirrored point falls
  in, rather than reflecting the voxel with a second formula (issue #27). The voxel picks the
  streamline and the coordinate measures depth along it, so the two must land on the same
  voxel; deriving one from the other makes that structural. Inside a voxel this is exactly
  `z_size - 1 - z`, the volume projectors' `np.flip(volume, axis=2)` convention; the old
  `z_size - z` was one voxel out. One mask, taken from the coordinates, drives both — the
  two masks used to disagree throughout the midline voxel.
- `angle.find_closest_streamline` mirrors the same way, but with the plain `z_size - 1 - z`
  in both directions (issue #33) — the query voxel onto the left, and the streamline back
  onto the right. Both ends are voxel-to-voxel there, so the plain formula round-trips
  exactly and the coordinate-derived form of #27 is not needed; it would in fact be worse,
  returning a streamline from the neighbouring voxel for a query on a voxel boundary. Its
  midline test is `voxel[2] > (z_size - 1) / 2`, not `> z_size / 2`: for an even `z_size`
  the first right-hand voxel is `z_size / 2` itself, and the old test left it unreflected
  and so absent from the left-only reference, which reported it as outside isocortex.
- `metrics.measure_streamline_layer_thicknesses` re-implements path deduplication inline
  with a Python loop instead of calling `processing.remove_duplicate_voxels_from_paths`,
  which is vectorized and has zero callers in the repo. The layer-sorting hack right below
  it is flagged as such in its own comment (`metrics.py:76`).
- `.python-version` (pinning 3.9) is **gitignored**, so a fresh clone is not pinned — `uv sync`
  there resolves to whatever interpreter uv finds (3.13 in this checkout). Nothing tests 3.9.
- `tests/test_real_data.py` imports from `tests.mini_ccf` *after* a
  `pytest.importorskip("h5py")`, so the import carries a `# noqa: E402`. It is the only
  suppression in the repo; the rest of the tree is clean under the configured rules.
- `docs/source/reference/processing.rst` titles itself `ccf_streamlines.projection` — wrong
  module name in the heading.
- A stale, untracked `ccf_streamlines.egg-info/` sits in the repo root from the pre-uv
  setuptools build. It is not the build source; `uv_build` is (`pyproject.toml`).

<!-- codebase-summarizer: end; generated 2026-08-28, commit c91140a, 39 claims -->
