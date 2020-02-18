"""
    BPE-NMT Encoder

    Copyright (C) 2020 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.
"""

from setuptools import setup, find_packages


setup(
    name='bpe_nmt',
    version='0.1.0',
    description='Tensor2tensor compatible BPE encoder',
    author='Haukur Barri Simonarson',
    license='MIT',
    classifiers=[
    'Intended Audience :: Developers',
    'Topic :: Text Processing',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    include_package_data=True
)
