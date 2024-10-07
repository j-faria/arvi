<p align="center">
  <img width = "140" src="https://github.com/j-faria/arvi/blob/main/docs/logo/logo.png?raw=true"/>
</p>

This package sits alongside [DACE](https://dace.unige.ch/) to help with the
analysis of radial velocity datasets.  
It has been used within the ESPRESSO GTO program, and may be useful for other
surveys and instruments.


## Getting started

Install `arvi` using pip  

```sh
pip install arvi

# or
pip install arvi -U # to update
```


Then either directly import a given target

```py
from arvi import HD1234
```

or create an instance of the `RV` class

```py
from arvi import RV
s = RV('HD1234', instrument='ESPRESSO')
```

#### Current version

![PyPI - Version](https://img.shields.io/pypi/v/arvi)

#### Actions

[![Deploy docs](https://github.com/j-faria/arvi/actions/workflows/docs-gh-pages.yml/badge.svg)](https://github.com/j-faria/arvi/actions/workflows/docs-gh-pages.yml)
[![Install-Test](https://github.com/j-faria/arvi/actions/workflows/install.yml/badge.svg)](https://github.com/j-faria/arvi/actions/workflows/install.yml)
[![Upload Python Package](https://github.com/j-faria/arvi/actions/workflows/python-publish.yml/badge.svg)](https://github.com/j-faria/arvi/actions/workflows/python-publish.yml)
