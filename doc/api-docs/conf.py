# -*- coding: utf-8 -*-
#
# InVEST 3 documentation build configuration file, created by
# sphinx-quickstart on Wed Nov 12 11:08:28 2014.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import importlib
import itertools
import os
import pkgutil
import subprocess
import sys
import warnings

from sphinx.ext import apidoc
from unittest.mock import MagicMock


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
DOCS_SOURCE_DIR = os.path.dirname(__file__)
INVEST_ROOT_DIR = os.path.join(DOCS_SOURCE_DIR, '..', '..')
INVEST_SOURCE_DIR = os.path.join(INVEST_ROOT_DIR, 'src')
sys.path.insert(0, INVEST_SOURCE_DIR)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'InVEST'
copyright = '2020, The Natural Capital Project'


print('setting package version...')
# set the package version so that modules can import natcap.invest.__version__
subprocess.run(['python', 'setup.py', '--version'],
               cwd=os.path.join(INVEST_SOURCE_DIR, 'natcap', 'invest'))
print('after setting version')

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import setuptools_scm
_version = setuptools_scm.get_version(
    root=os.path.join(INVEST_SOURCE_DIR, 'natcap', 'invest'),
    version_scheme='post-release',
    local_scheme='node-and-date'
)
print('_version:', _version)
version = _version.split('+')[0]
# The full version, including alpha/beta/rc tags.
release = _version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, keep warnings as "system message" paragraphs in the built documents.
keep_warnings = False
#keep_warnings = True


# -- Options for HTML output ----------------------------------------------

import sphinx_rtd_theme
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/invest-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.gif"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'InVESTdoc'

# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'InVEST.tex', 'InVEST Documentation',
   'The Natural Capital Project', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'invest', 'InVEST Documentation',
     ['The Natural Capital Project'], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'InVEST', 'InVEST Documentation',
   'The Natural Capital Project', 'InVEST', 
   'Integrated Valuation of Ecosystem Services and Tradeoffs',
   'Scientific Software'),
]


# -- Prepare for sphinx build ---------------------------------------------

# build cython extensions in-place so that sphinx can find the .so files
# alongside the source code
subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], 
               cwd=INVEST_ROOT_DIR)


# As suggested here https://stackoverflow.com/questions/27325165/metaclass-error-when-extending-scipy-stats-rv-continuous-mocked-for-read-the-doc
# Classes involved in multiple inheritance from a mocked class: 
#   * Container(QtWidgets.QGroupBox, InVESTModelInput) 
#   * Executor(QtCore.QObject, threading.Thread)
# We have to explicitly define the mocked classes so that `type(mocked class)` 
# is `type` and not `unittest.mock.MagicMock` to avoid metaclass conflict error

# Because Container inherits from QtWidgets.QGroupBox and InVESTModelInput
# which both are mocked, we have to give them separate classes
# Otherwise we get an MRO error
class MockQObject:
  pass
class MockQGroupBox:
  pass
mock_qtpy = MagicMock()
mock_qtpy.QtCore.QObject = MockQObject
mock_qtpy.QtWidgets.QGroupBox = MockQGroupBox
sys.modules.update([
  ('qtpy', mock_qtpy),
  ('qtpy.QtCore', MagicMock()),
  ('qtpy.QtGui', MagicMock()),
  ('qtpy.QtWidgets', MagicMock())
])

# Use sphinx apidoc tool to generate documentation for invest. Generated rst 
# files go into the api/ directory. Note that some apidoc options may not work 
# the same because we aren't using their values in the custom templates
apidoc.main([
    '--force',  # overwrite any files from previous run
    '-o', os.path.join(DOCS_SOURCE_DIR, 'api'),  # output to api/
    '--templatedir', os.path.join(DOCS_SOURCE_DIR, 'templates'),  # use custom templates
    '--separate',  # make a separate page for each module
    '--no-toc',  # table of contents page is redundant
    INVEST_SOURCE_DIR
])


# -- Generate model entrypoints file --------------------------------------

MODEL_RST_TEMPLATE = """
.. _models:

=========================
InVEST Model Entry Points
=========================

All InVEST models share a consistent python API:

    1) The model has a function called ``execute`` that takes a single python
       dict (``"args"``) as its argument.
    2) This arguments dict contains an entry, ``'workspace_dir'``, which
       points to the folder on disk where all files created by the model
       should be saved.

Calling a model requires importing the model's execute function and then
calling the model with the correct parameters.  For example, if you were
to call the Carbon Storage and Sequestration model, your script might
include

.. code-block:: python

    import natcap.invest.carbon.carbon_combined
    args = {
        'workspace_dir': 'path/to/workspace'
        # Other arguments, as needed for Carbon.
    }

    natcap.invest.carbon.carbon_combined.execute(args)

For examples of scripts that could be created around a model run,
or multiple successive model runs, see :ref:`CreatingSamplePythonScripts`.


.. contents:: Available Models and Tools:
    :local:

"""

EXCLUDED_MODULES = [
    '_core',  # anything ending in '_core'
    'recmodel_server',
    'recmodel_workspace_fetcher'
]
MODEL_ENTRYPOINTS_FILE = os.path.join(DOCS_SOURCE_DIR, 'models.rst')

# Find all importable modules with an execute function
# write out to a file models.rst in the source directory
all_modules = {}
for _loader, name, _is_pkg in itertools.chain(
        pkgutil.walk_packages(path=[INVEST_SOURCE_DIR]),  # catch packages
        pkgutil.iter_modules(path=[INVEST_SOURCE_DIR])):  # catch modules

    if (any([name.endswith(x) for x in EXCLUDED_MODULES]) or
        name.startswith('natcap.invest.ui')):
        continue

    try:
        module = importlib.import_module(name)
    except Exception as ex:
        print(ex)
        continue

    if not hasattr(module, 'execute'):
        continue

    try:
        module_title = module.execute.__doc__.strip().split('\n')[0]
        if module_title.endswith('.'):
            module_title = module_title[:-1]
    except AttributeError:
        module_title = None
    all_modules[name] = module_title

# Write sphinx autodoc function for each entrypoint
with open(MODEL_ENTRYPOINTS_FILE, 'w') as models_rst:
    models_rst.write(MODEL_RST_TEMPLATE)
    for name, module_title in sorted(all_modules.items(),
                                     key=lambda x: x[1]):
        models_rst.write((
            '{module_title}\n'
            '{underline}\n'
            '.. autofunction:: {modname}.execute\n\n').format(
                module_title=module_title,
                underline=''.join(['=']*len(module_title)),
                modname=name
            )
        )
