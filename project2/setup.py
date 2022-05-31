from setuptools import setup, Extension

setup(
    name='mykmeanssp',
    version='1.0.0',
    description='A Python C-API that implements kmeans++ algorithm',
    ext_modules=[Extension('mykmeanssp', ['kmeans.c'])]
)