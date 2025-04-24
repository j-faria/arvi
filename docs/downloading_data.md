
**arvi** can use the DACE API to download data products for a specific target.  
Each `RV` instance has three distinct methods for downloading CCFs,
S1Ds, and S2Ds.

```py
from arvi import RV
```

```py
s = RV('HD69830')
```


```py
s.download_ccf()
s.download_s1d()
s.download_s2d()
```

By default, this will download the files into a folder called `[star]_downloads`.
The methods all share the same arguments and are documented below.

As an example, we can download the 5th S2D file from ESPRESSO19 into a specific
directory

```py
s.download_s1d(instrument='ESPRESSO19', index=5, directory='some_directory')
```



!!! note 

    Depending on your network, the downloads can be slow, especially when 
    downloading many large S2D files like those from ESPRESSO for example.


---

::: arvi.timeseries.RV.download_ccf
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: arvi.timeseries.RV.download_s1d
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: arvi.timeseries.RV.download_s2d
    handler: python
    options:
        show_root_heading: true
        show_source: false