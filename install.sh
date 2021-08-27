#!/bin/bash
# Copyright (c) 2021, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Usage:  source install.sh
#
# This script installs all the packages required to build DeepConsensus.
#
# This script will run as-is on Ubuntu 18.
#
# We also assume that apt-get is already installed and available.

# ------------------------------------------------------------------------------
# Global setting for nucleus builds
# ------------------------------------------------------------------------------

NUCLEUS_BAZEL_VERSION="3.1.0"
NUCLEUS_TENSORFLOW_VERSION="2.4.0"
NUCLEUS_PIP_VERSION=0.5.8

function note_build_stage {
  echo "========== [$(date)] Stage '${1}' starting"
}

# Update package list
################################################################################
note_build_stage "Update package list"
sudo -H apt-get -qq -y update

# Install generic dependencies
################################################################################
note_build_stage "Update misc. dependencies"
sudo -H apt-get -y install pkg-config zip g++ zlib1g-dev unzip curl git lsb-release

# <internal>
# Install htslib dependencies
################################################################################
note_build_stage "Install htslib dependencies"
sudo -H apt-get -y install libssl-dev libcurl4-openssl-dev liblz-dev libbz2-dev liblzma-dev

# Install pip
################################################################################
note_build_stage "Update pip"
sudo -H apt-get -y install python3-dev python3-pip python3-wheel python3-setuptools
sudo -H apt-get -y update
# TensorFlow 2.0 requires pip >= 19.0
python3 -m pip install --user -U "pip==20.1.1"

# Update PATH so that newly installed pip is the one we actually use.
export PATH="$HOME/.local/bin:$PATH"
echo "$(pip --version)"

# Install python packages used by DeepConsensus.
################################################################################
python3 -m pip install --user 'wheel>=0.36'
python3 -m pip install --user 'numpy>=1.19'
python3 -m pip install --user 'pandas>=1.1'
python3 -m pip install --user 'tensorflow==2.4.0'
python3 -m pip install --user 'tf-models-official==2.4.0'
python3 -m pip install --user 'ml_collections>=0.1.0'
# Use apache-beam==2.20.0 for compatible protobuf version:
# https://github.com/apache/beam/blob/release-2.20.0/sdks/python/container/base_image_requirements.txt#L38
python3 -m pip install --user 'apache-beam==2.20.0'
python3 -m pip install --user 'apache-beam[test]==2.20.0'
python3 -m pip install --user "google-nucleus==${NUCLEUS_PIP_VERSION}"
python3 -m pip install --user 'absl-py>=0.13.0'

# We need a recent version of setuptools, because pkg_resources is included in
# setuptools, and protobuf's __init__.py contains the line
# __import__('pkg_resources').declare_namespace(__name__)
# and only recent versions of setuptools correctly sort the namespace
# module's __path__ list when declare_namespace is called.
python3 -m pip install --user 'setuptools>=49.6.0'

# Pre-compile the proto file.
mkdir -p nucleus/protos
curl "https://raw.githubusercontent.com/google/nucleus/${NUCLEUS_PIP_VERSION}/nucleus/protos/bed.proto" \
 > nucleus/protos/bed.proto

sudo apt install -y protobuf-compiler
protoc deepconsensus/protos/deepconsensus.proto --python_out=.
