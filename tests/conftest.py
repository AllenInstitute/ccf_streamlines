"""Shared fixtures.

The mini-CCF fixtures build a miniature but structurally faithful atlas into a
temporary directory, so the projector and boundary classes can be constructed
through their real public interface with no large data files and no mocking.
See ``tests/mini_ccf.py`` for the format and the reasoning.

The ``real_data`` marker gates a second, opt-in tier that runs the same public
interfaces against the actual assets. It never runs in CI: it is skipped unless
``CCF_STREAMLINES_TEST_DATA`` names a directory holding them.
"""

import os
from pathlib import Path

import pytest

from tests.mini_ccf import build_mini_ccf, reference_layer_thicknesses

REAL_DATA_ENV_VAR = "CCF_STREAMLINES_TEST_DATA"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: needs the real CCF assets; skipped unless "
        f"{REAL_DATA_ENV_VAR} names the directory holding them.",
    )


@pytest.fixture
def mini_ccf_factory(tmp_path):
    """Build a mini-CCF with structural properties perturbed one at a time.

    Returns a callable taking the same keyword arguments as
    ``mini_ccf.build_mini_ccf``. Each call writes into its own subdirectory, so
    a single test can build several variants.
    """
    counter = {"n": 0}

    def build(**overrides):
        counter["n"] += 1
        root = tmp_path / f"mini_ccf_{counter['n']}"
        return build_mini_ccf(root, **overrides)

    return build


@pytest.fixture
def mini_ccf(mini_ccf_factory):
    """A mini-CCF built with the default dimensions."""
    return mini_ccf_factory()


@pytest.fixture
def layer_thicknesses():
    """The reference layer thickness dict, as ``guide.rst`` builds it."""
    return reference_layer_thicknesses()


@pytest.fixture(scope="session")
def real_data_dir():
    """Directory holding the real assets, or skip.

    The expected layout is the published one: ``streamlines/``, ``view_lookup/``,
    ``cortical_metrics/`` and ``master_updated/`` subdirectories.
    """
    value = os.environ.get(REAL_DATA_ENV_VAR)
    if not value:
        pytest.skip(
            f"set {REAL_DATA_ENV_VAR} to the directory holding the real CCF "
            f"assets to run the real-data tier"
        )
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{REAL_DATA_ENV_VAR}={value} is not a directory")
    return path
