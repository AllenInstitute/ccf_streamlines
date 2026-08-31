# 1. Test against a hand-built miniature atlas, not real or sampled data

Date: 2026-08-28

Status: accepted

## Context

Every entry point that matters in this package is a class whose constructor
reads hundreds of megabytes of HDF5: a surface-paths file of about 452 MiB, a
closest-surface-voxel lookup of 185 MiB, per-streamline layer thicknesses of
101 MiB. The minimum working set is roughly 800 MB.

None of that can be committed to the repository or downloaded on a CI runner,
and an external contributor cannot be expected to obtain it. So the code that
most needs testing - about 1,900 lines of numerical geometry, with two open bug
reports against it - is precisely the code that appears hardest to reach.

Before this decision the suite was two tests covering one function, no CI ran
on pull requests, and a memory-reduction commit had already introduced a crash
that reached users and was reported back by an external contributor.

The enabling observation is that the package needs no real data to be tested
thoroughly. It makes no network calls and holds no absolute paths in its
source: every asset is a caller-supplied file path. Files that are *structurally*
faithful are therefore indistinguishable, from the library's point of view,
from the real ones.

## Decision

A single factory (`tests/mini_ccf.py`) builds a coherent miniature atlas - a
**mini-CCF** - into a temporary directory per test: six files totalling roughly
20 KB. Tests construct the real classes from those paths.

The factory maintains every cross-file invariant simultaneously, and asserts
them at build time so an invalid override fails with a readable message instead
of a reshape error deep inside library code:

- streamline rows contain flat voxel indices valid in the miniature volume;
- the flat volume lookup maps each streamline's start voxel to its path index
  and holds the sentinel everywhere else;
- view-lookup entries reference surface voxels that are genuinely streamline
  starts, and include tied keys;
- closest-surface-voxel targets are streamline starts, in a strictly increasing
  first column;
- layer thickness rows are contiguous and sum to the streamline's own arc
  length plus one voxel.

Dimensions are constrained rather than chosen, and each is justified in the
module. The most constrained is the padded streamline length: layer-normalized
slab projection rounds per-layer blocks against each layer's fraction of the
total and reshapes the stack into a final dimension of that same length, so the
rounded blocks must sum exactly. Swept against the real layer ratios, only 11,
12, 16, 18, 20, 21, 22 and 24 work. Twelve is the default - eleven is the true
minimum with all six layers present, but twelve gives layer 4 two voxels rather
than one.

The format was established by **inspecting the real assets**. The repository's
notes on them described a superseded generation of the files and were wrong in
several particulars, so they were not trusted, and they have since been removed
rather than corrected — a prose note that nothing checks drifts again. Two
findings materially changed construction:

- the flat volume lookup is populated **only at streamline start voxels**
  (~0.12% fill), not along whole streamlines;
- total layer thickness equals arc length **plus one voxel**, so fixtures must
  scale layer thicknesses to the miniature arc length or every layer-normalized
  depth returns the missing-value sentinel.

The format description now lives in `tests/mini_ccf.py`, next to the code that
depends on it, and every claim it makes is asserted against the real files by
`tests/test_real_data.py`. A claim that stops being true fails a test.

## Consequences

The suite runs anywhere, in seconds, with no data files, so it can gate pull
requests on every push - including on an ARM64 runner, which is not optional,
because one of the two open defects exists *only* as a divergence between
processor architectures.

Because fixtures are miniature and the geometry is chosen deliberately,
assertions are exact integer equalities rather than tolerances. A query at a
streamline voxel corner has an offset residual of exactly zero, so a projected
view coordinate equals the expected pixel with no floating-point slack.

A test asserting how numpy's *default* sort orders tied keys is not portable,
even at fourteen elements. The first run of this matrix showed the default and
stable sorts of the fixture's view-lookup keys agreeing on the arm64 runners
and on a local x86-64 workstation, and disagreeing on GitHub's x86-64 runners.
Tests about sort stability therefore inject both orderings explicitly rather
than relying on the platform's default, and the matrix earned its place inside
a minute of first running.

Two costs are accepted:

**Fixtures can quietly become self-consistent fiction** - exactly as the format
documentation already did. This is mitigated by a second, opt-in tier
(`tests/test_real_data.py`) which runs the same public interfaces against the
real assets when `CCF_STREAMLINES_TEST_DATA` names them. It is skipped by
default and never runs in CI. Its purpose is drift detection, plus confirming
the stable-sort behaviour at genuine tie density, which is a scale property no
miniature fixture reproduces.

**Named view presets cannot be exercised at miniature scale.** Their offsets are
absolute counts sized for full-resolution views, and cropping a miniature view
by them empties it. These are covered by asserting the resolved attribute and
the error raised on an unknown name, rather than by a round trip.

## Alternatives considered

**In-memory alternate constructors taking arrays directly.** Rejected: this
introduces a second, lower seam that bypasses the loading code - and the
loading code is where a substantial share of the known defects live. The
existing constructor file-path arguments are already the highest available
seam, and they need no source change at all.

**Committing downsampled slices of the real assets.** Rejected: every flat
index would have to be renumbered against the smaller volume, which means
writing the same factory anyway, and it puts binaries in version control.

**Mocking h5py.** Rejected: it would assert that the code calls the library a
certain way rather than that it computes the right numbers, and it would not
have caught either open defect.

## One source change

Threading `chunk_size` through the two relevant constructors, defaulting to the
current hardcoded 1000. The chunked read that most recently regressed is
unreachable below roughly a thousand view rows, so a miniature fixture cannot
touch it otherwise. This places the seam at the same level as the primary one
rather than introducing a lower one, and is behaviour-preserving at the default.
