#!/bin/bash
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
# Script to test DeepConsensus.
# Tested with Python3.8.
# Before running this, run:
#   source install.sh

set -euo pipefail

python3 -m deepconsensus.inference.quick_inference_test
# TODO Keep only the quick inference test for now.
# python3 -m deepconsensus.models.data_providers_test
# python3 -m deepconsensus.models.losses_and_metrics_test
# python3 -m deepconsensus.models.model_inference_transforms_test
# python3 -m deepconsensus.models.networks_test
# python3 -m deepconsensus.postprocess.stitch_utils_test
# python3 -m deepconsensus.preprocess.preprocess_test
# python3 -m deepconsensus.preprocess.utils_test
# python3 -m deepconsensus.utils.utils_test

# # These tests take longer to run.
# python3 -m deepconsensus.models.model_inference_test
# python3 -m deepconsensus.models.model_train_custom_loop_test
# python3 -m deepconsensus.models.model_utils_test
