"""Setup script for GenesisTasks package."""

from setuptools import find_packages, setup

# Read version from __init__.py or use default
__version__ = "0.1.0"

setup(
    name="genesis_assets",
    version="0.0.0",
    packages=find_packages(),
    author="Zaterval(interval-package) | ZiangZheng",
    maintainer="Ziang Zheng",
    maintainer_email="ziang_zheng@foxmail.com",
    url="https://github.com",
    license="BSD-3",
    description="RenforceRL",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
    ],
)
