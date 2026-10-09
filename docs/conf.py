"""Sphinx configuration for the exege guide, built on Read the Docs.

The pages are Markdown, read by MyST; the theme is Furo. Read the Docs builds HTML, a
PDF, an EPUB and a zipped single-page HTML from this file (`.readthedocs.yaml`).
"""

import re
from pathlib import Path

# The version is read rather than imported, so it is this checkout's even where an older
# exege is installed.
_init = (Path(__file__).parents[1] / "src" / "exege" / "__init__.py").read_text()
release = re.search(r'^__version__ = "(.+)"$', _init, re.MULTILINE)[1]
version = release

project = "exege"
author = "mahf708"
copyright = "2026, mahf708"
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_click",
    "sphinx_copybutton",
]

# The API reference is read from the code. torch is mocked rather than installed: Read the
# Docs has no use for a gigabyte of it to print docstrings.
autodoc_mock_imports = ["torch"]
autodoc_member_order = "bysource"
# Quotes and ellipses only: `--headless` in a docstring or a command's help stays two dashes.
smartquotes_action = "qe"
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# AGENTS.md files are contracts for contributors, not published pages.
exclude_patterns = ["AGENTS.md", "_build"]

# `[text](latents.md#python-api)` links reach headings down to ####.
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "deflist"]

# `console` blocks copy without their `$ ` prompts or their output, and a command
# continued with a trailing `\` copies whole.
copybutton_prompt_text = "$ "
copybutton_only_copy_prompt_lines = True
copybutton_line_continuation_character = "\\"

# -- HTML --------------------------------------------------------------------

html_theme = "furo"
html_title = "exege documentation"
html_theme_options = {
    "source_repository": "https://github.com/mahf708/exege",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- PDF ---------------------------------------------------------------------

# xelatex, for the em dashes and the ē the pages are written with.
latex_engine = "xelatex"
latex_documents = [("index", "exege.tex", "exege", author, "manual")]
latex_elements = {"papersize": "letterpaper", "pointsize": "10pt"}

# -- EPUB --------------------------------------------------------------------

epub_basename = "exege"
epub_show_urls = "no"
