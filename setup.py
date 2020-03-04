# This file is part of ganpdfs

import sys
import os
import re
from setuptools import setup, find_packages

PACKAGE = "wganpdfs"

# TODO: reduce the amount of requirements
requirements = ["hyperopt", "numpy", "keras", "scipy", "matplotlib"]

if sys.version_info < (3,6):
    print("cyclejet requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)


with open('README.md') as f:
    long_desc = f.read()

def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join('src', PACKAGE, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

setup(name=PACKAGE,
      version=get_version(),
      description = "WGAN models for PDFs",
      author = "",
      author_email = "",
      url="https://github.com/N3PDF/ganpdfs",
      long_description = long_desc,
      install_requires = requirements,
      entry_points = {'console_scripts':
                      ['wganpdfs = wganpdfs.run:main',]},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )

try:
    import lhapdf
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of LHAPDF")
