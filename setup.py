# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Builds the DeepConsensus package.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README_pip.md').read_text(encoding='utf-8')

REQUIREMENTS = (here / 'requirements.txt').read_text().splitlines()
EXTRA_REQUIREMENTS = {
    'cpu': ['intel-tensorflow>=2.4.0,<=2.7.0'],
    'gpu': ['tensorflow-gpu>=2.4.0,<=2.7.0']
}


def get_version():
  """Fetch version from utils/dc_constants.py."""
  with open(here / 'deepconsensus/utils/dc_constants.py', 'r') as constants:
    for line in constants:
      if line.startswith('__version__'):
        return line.split('=')[1].strip(' \'\n')


setup(
    # To support installation via
    #
    # $ pip install deepconsensus
    name='deepconsensus',
    version=get_version(),  # Retrieved from dc_constants.__version__.
    description='DeepConsensus',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/deepconsensus',
    author='Google LLC',
    keywords='bioinformatics',
    packages=find_packages(where='.'),
    package_dir={'deepconsensus': 'deepconsensus'},
    python_requires='>=3.6',
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    entry_points="""
    [console_scripts]
    deepconsensus = deepconsensus.cli:run
    """,
)
