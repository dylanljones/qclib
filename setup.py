# coding: utf-8
#
# This code is part of qclib.
#
# Copyright (c) 2021, Dylan Jones

from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name='qclib',
    version='0.0.0',
    author='Dylan Jones',
    author_email='dylanljones94@gmail.com',
    description='Collection of quantum algorithms.',
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/dylanljones/qclib',
    license='MIT License',
    packages=find_packages(),
    install_requires=requirements(),
    python_requires='>=3.6',
)
