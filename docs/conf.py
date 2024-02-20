import os
import sys

# -- General configuration ------------------------------------------------
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    # 'sphinxcontrib.katex',
    'sphinx_autodoc_typehints',
    # 'coverage_check',
    # 'myst_nb',  # This is used for the .ipynb notebooks
    # 'sphinx_gallery.gen_gallery',
    # 'sphinxcontrib.collections'
]

html_theme = 'sphinx_book_theme'

# html_static_path = ['_static']
