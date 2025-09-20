# `aerosol-functions` package

This package is a collection of tools to analyze and visualize atmospheric aerosol and ion data.

Latest version: 0.1.14

## Installation

Install directly from GitHub using `pip`

```bash
pip install git+https://github.com/jlpl/aerosol-functions.git
```

Install from PYPI:

```bash
pip install aerosol-functions
```

## Documentation

The package's documentation page can be found [here](https://jlpl.github.io/aerosol-functions/)

## Example 

Calculate the coagulation coefficient in cubic meters per second for 10 nm and 100 nm aerosol particles under standard conditions.

```python
import aerosol.functions as af

K = af.coagulation_coef(10e-9,100e-9)

print(K) # output: 2.4e-14
```
