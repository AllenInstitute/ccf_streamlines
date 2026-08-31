"""The layer key list, which has one definition shared by three readers.

``Isocortex3dProjector`` and ``IsocortexCoordinateProjector`` expose it as a
class attribute and ``metrics`` keys ``LAYER_LABELS`` on it. The three must
agree, because a thickness file written by ``metrics`` is read back by the
projectors -- so they share ``projection.ISOCORTEX_LAYER_KEYS`` rather than
each repeating the six names.
"""

from ccf_streamlines import metrics
from ccf_streamlines.projection import (
    ISOCORTEX_LAYER_KEYS,
    Isocortex3dProjector,
    IsocortexCoordinateProjector,
)


def _metrics_layer_names():
    """The layer names `measure_streamline_layer_thicknesses` writes."""
    return list(metrics.LAYER_LABELS.values())


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


def test_the_layer_keys_have_a_single_definition():
    """Equal values are not enough: independent literals can drift.

    Sharing one definition makes the agreement structural rather than
    coincidental.
    """
    assert (
        Isocortex3dProjector.ISOCORTEX_LAYER_KEYS
        is IsocortexCoordinateProjector.ISOCORTEX_LAYER_KEYS
        is ISOCORTEX_LAYER_KEYS
    )


def test_metrics_keys_its_labels_on_the_shared_definition():
    """The ontology IDs are metrics' own; the names it pairs them with are not."""
    assert list(metrics.LAYER_LABELS.values()) is not ISOCORTEX_LAYER_KEYS
    assert list(metrics.LAYER_LABELS.values()) == ISOCORTEX_LAYER_KEYS
    assert len(metrics.ISOCORTEX_LAYER_STRUCTURE_SET_IDS) == len(ISOCORTEX_LAYER_KEYS)
