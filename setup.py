from setuptools import setup

with open("README.md","r") as fh: 
    long_description = fh.read()

setup(
    name="aerosol-functions",
    version="0.0.8",
    description='Functions to analyze atmospheric aerosol data',
    package_dir={'':'src'},
    packages=['aerosol'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.9',
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
