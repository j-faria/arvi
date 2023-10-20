# Welcome to the `arvi` documentation

This site hosts the documentation for the 
**A**utomated **RV** **I**nspector, or `arvi` for short.

<figure markdown>
  ![Image title](logo/detective.png){ width="100" }
</figure>

This package sits alongside [DACE](https://dace.unige.ch/) to help with the
analysis of radial velocity datasets.  
It has been used extensively for data analysis of the ESPRESSO GTO program, and
may be useful for other surveys and instruments.


## Getting started

Install `arvi` using pip

```sh
pip install arvi
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


Read more in the [detailed analysis](detailed) page.



