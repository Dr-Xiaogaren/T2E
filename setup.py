#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import setuptools

setup(
    name="T2E",  # Replace with your own username
    version="0.0.1",
    description="benchmark for multi-robot target trapping",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="xiaogaren",
    author_email="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=" ",
    python_requires='>=3.6',
)
