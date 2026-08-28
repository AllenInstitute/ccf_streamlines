# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import re

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


project = 'ccf_streamlines'
copyright = '2022, Nathan Gouwens'
author = 'Nathan Gouwens'

# Single-sourced from pyproject.toml so the docs cannot drift from the package
# version. Read directly rather than via importlib.metadata, since Read the Docs
# builds the docs without installing the project.
with open(os.path.join(_REPO_ROOT, "pyproject.toml")) as _f:
    release = re.search(r'^version = "([^"]+)"', _f.read(), re.MULTILINE).group(1)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["h5py", "numpy", "scipy", "pandas", "nrrd", "tqdm", "skimage"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
