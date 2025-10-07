import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spark'
copyright = '2025, Mario Franco'
author = 'Mario Franco'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'autoapi.extension',
    'sphinx_design',
]

# General
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.ipynb', '.md']
master_doc = 'index'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Auto API
autoapi_dirs = ['../spark']
autoapi_file_patterns = ['*.py', '*.pyi']
autoapi_generate_api_docs = True
autoapi_type = 'python'
autoapi_template_dir = '_templates/autoapi'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]
autoapi_own_page_level = 'module'
autoapi_python_use_all = True
autoapi_ignore = ['*/tests/*', '*.env*', '*Typevar*', '*Generic*', '*ConfigT*', '*InputT*']
typehints_fully_qualified = False
typehints_use_rtype = False
autodoc_typehints = 'none'
autosummary_generate = True

# Sphinx Design
sd_custom_directives = {
    "dropdown-attributes": {
        "inherit": "dropdown",
        "argument": "Attributes",
        "options": {
            "color": "primary",
            "icon": "book",
        },
    }
}


