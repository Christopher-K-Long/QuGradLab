# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuGradLab'
copyright = '2025, Christopher K. Long'
author = 'Christopher K. Long'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.duration',
              'sphinx.ext.mathjax',
              'sphinx_math_dollar',
              'myst_parser',
              'stubdoc',
              'sphinx_tabs.tabs',
              'sphinx_codeautolink',
              'sphinx.ext.intersphinx']
myst_enable_extensions = ["dollarmath"]

autosummary_generate = True # Turn on sphinx.ext.autosummary
autosummary_imported_members = True
# autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
viewcode_line_numbers = True
myst_heading_anchors = 5
suppress_warnings = ["myst.header"]
module_names = ["qugradlab.systems.semiconducting.esr._controls",
                "qugradlab.pulses.invertible_functions.scaling",
                "qugradlab.pulses.invertible_functions.packaging",
                "qugrad.systems._systems"]
sphinx_tabs_disable_tab_closing = True
toc_object_entries_show_parents = 'hide'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

from docutils.nodes import literal, math
from docutils.nodes import  doctest_block, image, literal_block, math_block, pending, raw, rubric, substitution_definition, target
math_dollar_node_blacklist = (literal, math, doctest_block, image, literal_block, math_block, pending, raw, rubric, substitution_definition, target)

mathjax3_config = {
    'loader': {'load': ['[tex]/mathtools', '[tex]/physics']},
    'tex': {
        'packages': {'[+]': ['mathtools', 'physics']},
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}

intersphinx_mapping = {
    'python': (
        "https://docs.python.org/3",
        None
    ),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    "py_ste": (
        "https://PySTE.readthedocs.io/en/latest",
        None,
    ),
    "qugrad": (
        "https://qugrad.readthedocs.io/en/latest",
        None,
    )
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))


DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_call(f"cd {DIR}; cffconvert -f apalike > docs/citation/citation_files/citation.txt", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f bibtex > docs/citation/citation_files/citation.bib", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f ris > docs/citation/citation_files/citation.ris", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f codemeta > docs/citation/citation_files/citation_codemeta.json", shell=True)
subprocess.check_call(f"cd {DIR}; cffconvert -f endnote > docs/citation/citation_files/citation.enw", shell=True)

def shorten_autosummary_titles(app, *args) -> None:
    """Remove module and class from the autosummary titles."""
    autosummary_dir = os.path.join(app.srcdir, "reference", "_autosummary")
    if not os.path.exists(autosummary_dir):
        return

    for filename in os.listdir(autosummary_dir):
        if not filename.endswith(".rst"):
            continue

        path = os.path.join(autosummary_dir, filename)
        with open(path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            continue
        
        for i, line in enumerate(lines):
            if len(line.strip()) != 0:
                break
            
        if line.count(".") < 2:
            continue

        short = line.strip().rsplit(".", 1)[-1]
        lines[i] = short + "\n"
        lines[i+1] = "=" * len(short) + "\n"
        with open(path, "w") as f:
            f.writelines(lines)

def setup(app) -> None:
    app.connect("env-before-read-docs", shorten_autosummary_titles)