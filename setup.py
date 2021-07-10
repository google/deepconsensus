# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Builds the DeepConsensus package.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import pathlib
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # To support installation via
    #
    # $ pip install deepconsensus
    name='deepconsensus',
    version='0.1.0',  # Keep in sync with __init__.__version__.
    description='DeepConsensus',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='TODO',
    author='Google LLC',
    keywords='TODO',
    packages=['deepconsensus'],
    package_dir={'deepconsensus': 'deepconsensus'},
    python_requires='==3.6',
    install_requires=[],
)
