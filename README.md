![logo](https://github.com/jlpl/aerosol-functions/raw/master/logo.png)

A collection of tools to analyze and visualize atmospheric aerosol and ion data.

## Installation
```shell
pip install aerosol-functions
```

## Documentation
See [here](https://jlpl.github.io/aerosol-functions/)

## Using the `aerosol-analyzer`
The `aerosol-analyzer` is a GUI application where one can visualize and analyze aerosol number size distribution data. The application runs in the browser using a bokeh server.

A general workflow is roughly the following:
1. Load aerosol number size distribution data from a CSV file. Note that it is recommended to first combine all the data into a single file before opening it in the `aerosol-analyzer`. In the CSV file:
    - First column: timestamps (e.g. in the format YYYY-MM-DD HH:MM:SS)
    - First row: particle diameters representing each size bin
    - The rest of the data should contain the number size distribution
2. Draw regions of interest (ROIs) on the number size distribution using the `FreehandDrawTool`
3. Select ROIs using the `TapTool` and do selected calculations on the data inside the ROIs. For example one can:
    - Fit (mixture) log-normal distributions on the selected size distributions
4. Save the ROIs including the calculated quantities to a JSON formatted file.
5. Continue working on a project by loading an already saved ROI file.

To start the application type on the command line:
```
aerosol-analyzer
```

Below is a screenshot from the application
