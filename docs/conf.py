# -- Path setup --------------------------------------------------------------

import os
import sys

TYPE_REWRITES = [
    ('~optax._src.base.GradientTransformation', 'optax.GradientTransformation'),
    ('~optax._src.base.Params', 'optax.Params'),
    ('~optax._src.base.Updates', 'optax.Updates'),
    ('~optax._src.base.OptState', 'optax.OptState'),
    ('base.GradientTransformation', 'optax.GradientTransformation'),
    ('base.Params', 'optax.Params'),
    ('base.Updates', 'optax.Updates'),
    ('base.OptState', 'optax.OptState'),
]


def _add_annotations_import(path):
  """Appends a future annotations import to the file at the given path."""
  with open(path) as f:
    contents = f.read()
  if contents.startswith('from __future__ import annotations'):
    # If we run sphinx multiple times then we will append the future import
    # multiple times too.
    return

  assert contents.startswith('#'), (path, contents.split('\n')[0])
  with open(path, 'w') as f:
    # NOTE: This is subtle and not unit tested, we're prefixing the first line
    # in each Python file with this future import. It is important to prefix
    # not insert a newline such that source code locations are accurate (we link
    # to GitHub). The assertion above ensures that the first line in the file is
    # a comment so it is safe to prefix it.
    f.write('from __future__ import annotations  ')
    f.write(contents)


def _recursive_add_annotations_import():
  for path, _, files in os.walk('../optax/'):
    for file in files:
      if file.endswith('.py'):
        _add_annotations_import(os.path.abspath(os.path.join(path, file)))


def _monkey_patch_doc_strings():
  """Rewrite function signatures to match the public API.

  This is a bit of a dirty hack, but it helps ensure that the public-facing
  docs have the correct type names and crosslinks.

  Since all optax code lives in a `_src` directory, and since all function
  annotations use types within that private directory, the public facing
  annotations are given relative to private paths.

  This means that the normal documentation generation process does not give
  the correct import paths, and the paths it does give cannot cross link to
  other parts of the documentation.

  Do we really need to use the _src structure for optax?

  Note, class members are not fixed by this patch, only function
    parameters. We should find a way to genearlize this solution.
  """
  import sphinx_autodoc_typehints
  original_process_docstring = sphinx_autodoc_typehints.process_docstring

  def new_process_docstring(app, what, name, obj, options, lines):
    result = original_process_docstring(app, what, name, obj, options, lines)

    for i in range(len(lines)):
      l = lines[i]
      for before, after in TYPE_REWRITES:
        l = l.replace(before, after)
      lines[i] = l

    return result

  sphinx_autodoc_typehints.process_docstring = new_process_docstring


_recursive_add_annotations_import()
_monkey_patch_doc_strings()


sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('../sketchyopts/'))

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

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}

autosummary_generate = True

apidoc_module_dir = '../sketchyopts/'
apidoc_output_dir = 'api'
# # apidoc_excluded_paths = ['../tests']
apidoc_separate_modules = True
# apidoc_module_first = True

python_use_unqualified_type_names = False
add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'

html_static_path = ['_static']

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/leiweimu/workflow_test',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}
