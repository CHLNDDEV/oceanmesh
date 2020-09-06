#!/usr/bin/env python

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 0):
    print("Python 3.0 or higher required, please upgrade.")
    sys.exit(1)


setup(
    name="pyOceanMesh",
    version="0.1.0",
    description="Automatic coastal ocean mesh generator",
    author="Keith J. Roberts",
    author_email="keithrbt0@gmail.com",
    url="https://github.com/CHLNDDEV/pyOceanMesh",
    packages=["pyOceanMesh"],
    install_requires=[
        "numpy",
        "scipy",
        "netcdf4",
        "pyshp",
        "matplotlib",
        "Pillow",
        "scikit-fmm",
    ],
)
