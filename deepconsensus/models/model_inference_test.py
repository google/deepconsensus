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
"""Tests for deepconsensus.models.model_inference."""

import glob
import os

from absl.testing import absltest

from deepconsensus.models import model_configs
from deepconsensus.models import model_inference
from deepconsensus.models import model_utils
from deepconsensus.utils import test_utils


class ModelInferenceTest(absltest.TestCase):

  def test_inference_e2e(self):
    """Tests that inference finishes running and an output file is created."""

    config_name = 'transformer_learn_values+test'
    out_dir = self.create_tempdir().full_path
    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    params = model_configs.get_config(config_name)
    tpu = None
    tpu_topology = None
    model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
    model_inference.run_inference(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        tpu=None,
        tpu_topology=None,
        limit=-1)

    # Output directory should contain the CSV logger output.
    csv_output = glob.glob(os.path.join(out_dir, 'inference.csv'))
    self.assertLen(csv_output, 1)


if __name__ == '__main__':
  absltest.main()
