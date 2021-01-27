"""
Generating synthetic Monte Carlo (MC) Parton distribution Function (PDF) replicas
using Generative Adversarial Networks (GANs).


This program has been developed within the N3PDF group. (n3pdf.mi.infn.it/)

Documentation:
https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/introduction.html

Authors: - Stefano Carrazza
         - Juan E. Cruz-Martinez
         - Tanjona R. Rabemananjara

License: GPL-3.0, 2020
"""


import pathlib
from setuptools import setup
from setuptools import find_packages

PACKAGE = "ganpdfs"
THIS_DIR = pathlib.Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIR / "README.md").read_text()
REQUIREMENTS = (THIS_DIR / "requirements.txt").read_text()

try:
    import lhapdf
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of LHAPDF")


setup(
    name=PACKAGE,
    version='0.1.0',
    description="GANs for PDF replicas",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="N3PDF",
    author_email="tanjona.rabemananjara@mi.infn.it",
    license="GPL 3.0",
    url="https://github.com/N3PDF/ganpdfs",
    zip_safe=False,
    project_urls={
        "Documentation": "https://n3pdf.github.io/ganpdfs/",
        "Source": "https://github.com/N3PDF/ganpdfs"
    },
    entry_points={
        "console_scripts": [
            "ganpdfs = ganpdfs.scripts.main:main"
            "postgans = ganpdfs.scripts.postgans:main",
        ]
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    setup_requires=["wheel"],
    python_requires='>=3.6'
)
