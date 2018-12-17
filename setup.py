#/usr/bin/env python3

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
        name            = 'mod_name',
        version         = '0.1',
        description     = 'My Python Extension Module',
        include_dirs    = get_numpy_include_dirs(),
        ext_modules     =   [
                            Extension(
                                'mod_name',
                                sources         = ['src/mod_name_module.c', 'src/utilities.c'],
                                include_dirs    = ['include/'])
                            ]
      )

