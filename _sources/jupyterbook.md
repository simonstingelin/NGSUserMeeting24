---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Jupyter-Book

## Why Jupyter-Book?

```{admonition} A quote from the Jupyter-Book website:

Build beautiful, publication-quality books and documents from computational content.
```

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/basics/organize.html
:link-type: url
:class-header: bg-light

Text content ✏️
^^^

Structure books with text files and Jupyter Notebooks with minimal configuration.
:::

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/content/myst.html
:link-type: url
:class-header: bg-light

MyST Markdown ✨
^^^

Write MyST Markdown to create enriched documents with publication-quality features.

:::

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/content/executable/index.html
:link-type: url
:class-header: bg-light

Executable content 🔁
^^^

Execute notebook cells, store results, and insert outputs across pages.

:::

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/interactive/launchbuttons.html
:link-type: url
:class-header: bg-light

Live environments 🚀
^^^

Connect your book with Binder, JupyterHub, and other live environments
:::

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/publish/web.html
:link-type: url
:class-header: bg-light

Build and publish 🎁
^^^

Share your built books via web services and hosted websites.
:::

:::{grid-item-card}
:link: https://jupyterbook.org/en/stable/content/components.html
:link-type: url
:class-header: bg-light

UI components ⚡
^^^

Create interactive and web-native components and services.
:::

::::

<!-- - Much LaTeX functionality and 
- Cross references over multiple files
- Figures with captions
- Proofs, Theorems, etc
    - `pip install sphinx-proof`
- Results, `glue`
- The result looks like a script
- Markdown files can also be executed with `jupytext` -->

## NGSolve in Jupyter-Book

Provided that NGSolve is installed in the python environment, NGSolve can also be integrated in a Juptyter book.

In order to be able to use the **webgui** in a book, the `webgui.js` must be saved in the `./_static` directory. This can be done with the following script:

```{code-cell} ipython3
from webgui_jupyter_widgets.js import widget_code 

fp = open("_static/webgui.js", "w")
fp.write(widget_code)
fp.close()
```

