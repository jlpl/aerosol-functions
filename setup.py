from setuptools import setup

with open("README.md","r") as fh: 
    long_description = fh.read()

setup(
    name="aerosol-functions",
    version="0.0.1",
    description='Functions to analyze atmospheric aerosol data',
    py_modules=["aerosol_functions"],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        "pandas >= 1.1.0",
        "numpy >= 1.19.0",
        "matplotlib >= 3.3.4",
        "scipy >= 1.5.3",
    ],
    url="https://github.com/jlpl/aerosol-functions",
    author="Janne Lampilahti",
    author_email="janne.lampilahti@helsinki.fi",
)
