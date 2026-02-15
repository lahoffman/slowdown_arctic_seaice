"""
Setup script for the slowdown_arctic_seaice package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Arctic sea ice slowdown analysis package"

setup(
    name="slowdown_arctic_seaice",
    version="0.1.0",
    author="Lauren Hoffman",
    author_email="lhoffma2@ucsc.edu",
    description="Modular package for analyzing Arctic sea ice trends and climate indices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lahoffman/slowdown_arctic_seaice",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.7",
        "openpyxl>=3.0.0",  # For Excel files
        "matplotlib>=3.4.0",
        "cartopy>=0.20.0",
        "cmocean>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "ipython>=7.0",
            "jupyter>=1.0",
        ],
    },
)
