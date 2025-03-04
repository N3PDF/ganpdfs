# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from recommonmark.transform import AutoStructify
sys.path.insert(0, os.path.abspath('..'))
# Used to determine the package version
import ganpdfs


# -- Project information -----------------------------------------------------

project = 'ganpdfs'
copyright = '2020, N3PDF'
author = 'Tanjona R. Rabemananjara, Juan Cruz-Martinez, Stefano Carrazza'

# The full version, including alpha/beta/rc tags
release = ganpdfs.__version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'sphinxcontrib.bibtex',
]
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Markdown configuration

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autosectionlabel_prefix_document = True
# Allow to embed rst syntax in  markdown files.
enable_eval_rst = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# Choose theme
# rtd theme defined here https://github.com/readthedocs/sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Intersphinx  -------------------------------------------------------------

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Doctest ------------------------------------------------------------------

doctest_path = []

# -- Autodoc ------------------------------------------------------------------

autodoc_member_order = 'bysource'

# Adapted this from
# https://github.com/readthedocs/recommonmark/blob/ddd56e7717e9745f11300059e4268e204138a6b1/docs/conf.py
# app setup hook
def setup(app):
    app.add_config_value(
            'recommonmark_config', {'enable_eval_rst': True, }, True
    )
    app.add_transform(AutoStructify)
