# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc - All Rights Reserved
#
# This file is part of geostackai
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#
# https://github.com/geo-stack/geostackai
# =============================================================================

"""Installation script """

import setuptools
from setuptools import setup
from geostackai import __version__
import csv


DESCRIPTION = "Library to assist AI projects at geostack."

with open('requirements.txt', 'r') as csvfile:
    INSTALL_REQUIRES = list(csv.reader(csvfile))
INSTALL_REQUIRES = [item for sublist in INSTALL_REQUIRES for item in sublist]

EXTRAS_REQUIRE = {
    'dev': [""]
    }

setup(name='geostackai',
      version=__version__,
      description=DESCRIPTION,
      license='Proprietary',
      author='Jean-Sébastien Gosselin',
      author_email='jsgosselin@geostack.ca',
      ext_modules=[],
      packages=setuptools.find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      classifiers=["Programming Language :: Python :: 3",
                   "License :: Proprietary",
                   "Operating System :: OS Independent"],
      )
