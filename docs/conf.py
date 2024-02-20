# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

# -- General configuration ---------------------------------------------------

master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    # 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.apidoc',
    # 'sphinxcontrib.katex',
    # 'sphinx_autodoc_typehints',
    # 'coverage_check',
    # 'myst_nb',  # This is used for the .ipynb notebooks
    # 'sphinx_gallery.gen_gallery',
    # 'sphinxcontrib.collections'
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

apidoc_module_dir = '../sketchyopts'
apidoc_output_dir = 'api'
apidoc_excluded_paths = ['../tests']
apidoc_separate_modules = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'

html_static_path = ['_static']
