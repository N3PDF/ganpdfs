# This file is part of ganpdfs

from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    print("cyclejet requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

with open('README.md') as f:
    long_desc = f.read()

setup(name= "wganpdfs",
      version = '0.0.1',  
      description = "WGAN models for PDFs",
      author = "",
      author_email = "",
      url="https://gitlab.cern.ch/N3PDF/wganpdfs",
      long_description = long_desc,
      entry_points = {'console_scripts':
                      ['wganpdfs = wganpdfs.run:main',]},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )
