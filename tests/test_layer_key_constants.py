"""The layer key list, which is written out three times.

``Isocortex3dProjector`` and ``IsocortexCoordinateProjector`` each carry their
own ``ISOCORTEX_LAYER_KEYS`` list literal, and ``metrics`` has the same six
names again in its ``LAYER_LABELS`` dict. The three must agree, because a
thickness file written by ``metrics`` is read back by the projectors -- but
nothing makes them agree, so editing one is a silent way to break the others.
"""

import inspect

import pytest

from ccf_streamlines import metrics
from ccf_streamlines.projection import (
    Isocortex3dProjector,
    IsocortexCoordinateProjector,
)

DUPLICATED_CONSTANT = (
    "the layer key list is duplicated across Isocortex3dProjector, "
    "IsocortexCoordinateProjector and metrics rather than shared; remove this "
    "marker when they refer to one definition"
)


def _metrics_layer_names():
    """The layer names `measure_streamline_layer_thicknesses` writes.

    They live in a dict local to the function, so they have to be read out of
    its source rather than imported -- which is itself the problem.
    """
    source = inspect.getsource(metrics.measure_streamline_layer_thicknesses)
    namespace = {}
    start = source.index("LAYER_LABELS")
    end = source.index("}", start) + 1
    exec(source[start:end].strip(), namespace)  # noqa: S102 - reading our own source
    return list(namespace["LAYER_LABELS"].values())


def test_the_two_projectors_list_the_same_layer_keys():
    assert (
        Isocortex3dProjector.ISOCORTEX_LAYER_KEYS
        == IsocortexCoordinateProjector.ISOCORTEX_LAYER_KEYS
    )


def test_metrics_writes_the_keys_the_projectors_read():
    """A thickness file from `metrics` must be loadable by the projectors."""
    assert _metrics_layer_names() == Isocortex3dProjector.ISOCORTEX_LAYER_KEYS


def test_the_keys_match_the_real_files(mini_ccf):
    from tests.mini_ccf import LAYER_KEYS

    assert list(LAYER_KEYS) == Isocortex3dProjector.ISOCORTEX_LAYER_KEYS


@pytest.mark.xfail(strict=True, reason=DUPLICATED_CONSTANT)
def test_the_layer_keys_have_a_single_definition():
    """Equal values are not enough: three independent literals can drift.

    They agree today, so this is not a live bug -- it is the reason one will
    appear. Sharing one definition makes the agreement structural.
    """
    assert (
        Isocortex3dProjector.ISOCORTEX_LAYER_KEYS
        is IsocortexCoordinateProjector.ISOCORTEX_LAYER_KEYS
    )
