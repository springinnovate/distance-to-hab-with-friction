"""setup.py module for shortest distances calculator."""
import numpy
from setuptools.extension import Extension
from setuptools import setup

setup(
    name='shortest_distances',
    maintainer='Rich Sharp',
    maintainer_email='richpsharp@gmail.com',
    packages=[
        'shortest_distances',
    ],
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'node-and-date'},
    include_package_data=True,
    license='BSD',
    zip_safe=False,
    ext_modules=[
        Extension(
            name="shortest_distances",
            sources=["shortest_distances.pyx"],
            include_dirs=[
                numpy.get_include(),
                ],
            language="c++",
        ),
    ]
)
