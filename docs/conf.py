import os
import sys


sys.path.insert(0, os.path.abspath('..'))

project = 'Optiland'
copyright = '2024, Kramer Harrison'
author = 'Kramer Harrison'
release = '0.1.5'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              'sphinx_gallery.gen_gallery']

autosummary_generate = True  # Automatically generate summaries

templates_path = ['_templates']
modindex_common_prefix = ['optiland.']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sphinx_gallery_conf = {
     'examples_dirs': 'examples',   # path to example scripts
     'gallery_dirs': 'auto_examples',  # gallery output directory
}

autodoc_mock_imports = ['numpy', 'yaml', 'scipy', 'matplotlib', 'numba',
                        'pandas']

pygments_style = 'sphinx'
