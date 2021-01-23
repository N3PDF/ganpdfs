"""
This file contains the current state of packaging in Python using
Distribution Utilities (Distutils) and its extension from the end
user'point-of-view.

Documentation:
https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/introduction.html

Authors: - Stefano Carrazza
         - Juan Cruz-Martinez
         - Tanjona R. Rabemananjara
"""


import os
import re
from setuptools import setup
from setuptools import find_packages

PACKAGE = "ganpdfs"

# Used for pytest and code coverage
TESTS_REQUIEREMENTS = [
        "pylint",
        "pytest",
        "pytest-cov",
        "pytest-env",
        "pygit2",
        "semver"
    ]

# Dependencies for the packages
PACKAGE_REQUIEREMENTS = [
        "tqdm",
        "hyperopt",
        "rich"
    ]

# Depending on the documents more dependencies can be added
DOCS_REQUIEREMENTS = [
        "recommonmark",
        "sphinx_rtd_theme",
        "sphinxcontrib-bibtex"
    ]

# Check if LHAPDF is installed
try:
    import lhapdf
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of LHAPDF")

# Read through Readme
try:
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    print("Read me file not found.")


def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


setup(
    name=PACKAGE,
    version=get_version(),
    description="GANs for PDF replicas",
    author="Stefano Carrazza, Juan Cruz-Martinez, Tanjona R. Rabemananjara",
    author_email="tanjona.rabemananjara@mi.infn.it",
    url="https://github.com/N3PDF/ganpdfs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=PACKAGE_REQUIEREMENTS,
    extras_require={"docs": DOCS_REQUIEREMENTS, "tests": TESTS_REQUIEREMENTS},
    entry_points={"console_scripts":
        [
            "ganpdfs = ganpdfs.scripts.main:main",
            "postgans = ganpdfs.scripts.postgans:main",
        ]
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    setup_requires=["wheel"],
    python_requires='>=3.6'
)
