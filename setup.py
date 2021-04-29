#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("jtool", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name='jtool',
    description="A Tool Library for JITTOR",
    url="https://agit.ai/Yi/jtool.git",
    version=get_version(),
    author="SongyiGao",
    author_email="songyigao@gmail.com",
    python_requires=">=3.7",
    install_requires=[
        "jittor",
        "numpy",
    ],
    
)