#!/bin/bash
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
#
# Script to install and build pip package for DeepConsensus.
# Tested with Python3.6.
# Before running this, run:
#   source install.sh

set -euo pipefail

# Test everything.

#### These tests passed:
python3 -m deepconsensus.models.data_providers_test
python3 -m deepconsensus.models.losses_and_metrics_test
python3 -m deepconsensus.models.majority_vote_transforms_test
python3 -m deepconsensus.models.model_inference_test
python3 -m deepconsensus.models.model_inference_transforms_test
python3 -m deepconsensus.models.model_inference_with_beam_test
python3 -m deepconsensus.models.model_train_custom_loop_test
python3 -m deepconsensus.models.model_utils_test
python3 -m deepconsensus.models.networks_test
python3 -m deepconsensus.models.run_majority_vote_model_test
python3 -m deepconsensus.postprocess.stitch_predictions_test
python3 -m deepconsensus.postprocess.stitch_predictions_transforms_test
python3 -m deepconsensus.preprocess.beam_io_test
python3 -m deepconsensus.preprocess.generate_input_test
python3 -m deepconsensus.preprocess.generate_input_transforms_test
python3 -m deepconsensus.preprocess.merge_datasets_test
python3 -m deepconsensus.preprocess.merge_datasets_transforms_test
python3 -m deepconsensus.preprocess.preprocess_utils_test
python3 -m deepconsensus.tf_examples.tf_example_transforms_test
python3 -m deepconsensus.tf_examples.tf_example_utils_test
python3 -m deepconsensus.tf_examples.write_tf_examples_test
python3 -m deepconsensus.utils.utils_test

# Build pip package for upload to PyPI.
pip3 install --upgrade twine
python3 setup.py sdist
python3 setup.py bdist_wheel
python3 -m twine check dist/*

# <internal>
# Remove the "--repository testpypi" section to push to the real PyPI.
# echo "python3 -m twine upload --repository testpypi dist/*"
