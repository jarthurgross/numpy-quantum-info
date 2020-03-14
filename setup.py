# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

install_requires = ['numpy']

setup(
    name='qinfo',
    version='0.1.0',
    description='Numpy/Scipy-like library for quantum information',
    long_description=readme,
    author='Jonathan A. Gross',
    author_email='jarthurgross@gmail.com',
    url='https://github.com/jarthurgross/numpy-quantum-information',
    license=license,
    install_requires=install_requires,
    packages=find_packages(exclude=('tests', 'docs'))
)
