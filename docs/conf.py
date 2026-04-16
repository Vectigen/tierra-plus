"""Sphinx configuration for Tierra+."""

project = "Tierra+"
author = "Vectigen"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
html_theme = "alabaster"
html_static_path = ["_static"]
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence"]
