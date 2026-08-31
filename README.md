# CCF streamlines

This package supports projection of Allen Common Coordinate Framework (CCF)
mouse brain volumetric data and coordinate data on to 2D representations. It
also enables other operations that make use of the "streamlines" between
the top and bottom of the isocortex in the CCF.


## Running the tests

The test suite needs no data files and finishes in seconds:

```
uv sync
uv run pytest
```

It builds a miniature but structurally faithful atlas into a temporary
directory and constructs the real classes from it, so contributors without
access to Allen Institute storage can run the whole suite. See
`docs/decisions/0001-mini-ccf-fixtures.md` for why.

A second, opt-in tier runs the same public interfaces against the real assets,
to catch the fixtures drifting from the files users actually download. It is
skipped unless you point an environment variable at them:

```
CCF_STREAMLINES_TEST_DATA=/path/to/assets uv run pytest -m real_data
```

Tests marked `xfail` are pinned to a known open bug or an open pull request.
They assert the *correct* behaviour, so the suite stays green while the bug
exists and reports an unexpected pass the moment it is fixed - which is the
signal to remove the marker.

## Level of support

This code is an important part of the internal Allen Institute code base and we are actively using and maintaining it. Issues are encouraged, but because this tool is so central to our mission pull requests might not be accepted if they conflict with our existing plans.
