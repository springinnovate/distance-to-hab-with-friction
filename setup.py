"""setup.py module for dist to hab w/ friction."""

import numpy
from setuptools.extension import Extension
from setuptools import setup

setup(
    name="shortest_distances",
    ext_modules=[
        Extension(
            name="shortest_distances",
            sources=["src/shortest_distances.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
        ],
)
