![logo](logo.png)

# Aerosol functions
A collection of functions to analyze and visualize atmospheric aerosol data.

Install using `pip install aerosol-functions`

See the [documentation](https://jlpl.github.io/aerosol-functions/)

## Example

An aerosol number size disitribution is assumed to be a `pandas.DataFrame`, denoted `df` where 

```
df.index:
    Time, pandas.DatetimeIndex
    
df.columns:
    Size bin diameters in meters, float
	
df.values:
    Normalized concentration dN/dlogDp in cm-3, float
```

One can for example calculate the number concentration in different size ranges

```python
import aerosol.functions as af
import numpy as np
import pandas as pd

dp_low_limits = np.array([3e-9,6e-9,10e-9])
dp_high_limits = np.array([6e-9,10e-9,20e-9])

conc_df = af.calc_conc(df,dp_low_limits,dp_high_limits)
```
