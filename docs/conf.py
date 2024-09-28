import os
import sys


def setup(app):
    app.add_css_file('custom.css')


sys.path.insert(0, os.path.abspath('..'))

project = 'Optiland'
copyright = '2024, Kramer Harrison'
author = 'Kramer Harrison'
release = '0.1.3'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              'sphinx_gallery.gen_gallery']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sphinx_gallery_conf = {
     'examples_dirs': 'examples',   # path to example scripts
     'gallery_dirs': 'auto_examples',  # gallery output directory
}
