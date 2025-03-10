from setuptools import setup

with open("README.md","r") as fh: 
    long_description = fh.read()

setup(
    name = "aerosol-functions",
    version = "0.1.12",
    description = 'Functions to analyze atmospheric aerosol data',
    package_dir = {'':'src'},
    packages = ['aerosol'],
    entry_points = {
        'console_scripts': [
            'aerosol-analyzer = aerosol.my_cmd:run_app',
        ],
    },
    long_description = long_description,
    long_description_content_type = "text/markdown",
    python_requires = '>=3.9',
    install_requires = [
        "pandas >= 1.1.0",
        "numpy >= 1.19.0",
        "matplotlib >= 3.3.4",
        "scipy >= 1.5.3",
        "bokeh >= 3.6.3",
        "astral >= 3.2",
        "kneed >= 0.8.5",
        "scikit-learn >= 1.3.0"
    ],
    url = "https://github.com/jlpl/aerosol-functions",
    author = "Janne Lampilahti",
    author_email = "janne.lampilahti@helsinki.fi",
)
