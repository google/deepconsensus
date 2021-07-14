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
"""Tests for deepconsensus.models.model_train_custom_loop."""

import glob
import os

from absl.testing import absltest
from absl.testing import parameterized

from deepconsensus.models import model_configs
from deepconsensus.models import model_train_custom_loop
from deepconsensus.models import model_utils


class ModelTrainTest(parameterized.TestCase):

  @parameterized.parameters(['fc+test', 'transformer+test'])
  def test_train_e2e(self, config_name):
    """Tests that training completes and output files written."""

    out_dir = self.create_tempdir().full_path
    params = model_configs.get_config(config_name)
    tpu = None
    tpu_topology = None
    model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
    model_train_custom_loop.train(
        out_dir=out_dir, params=params, tpu=tpu, tpu_topology=tpu_topology)

    # Output directory should contain TensorBoard event files for training and
    # eval, model checkpoint files.
    train_event_file = glob.glob(os.path.join(out_dir, 'train/*event*'))
    eval_event_file = glob.glob(os.path.join(out_dir, 'eval/*event*'))
    self.assertLen(train_event_file, 1)
    self.assertLen(eval_event_file, 1)
    checkpoint_files = glob.glob(os.path.join(out_dir, 'checkpoint*'))
    # +2 here for checkpoint and checkpoint_metrics.tsv
    self.assertLen(checkpoint_files, params.num_epochs * 2 + 2)
    json_params = glob.glob(os.path.join(out_dir, 'params.json'))
    self.assertLen(json_params, 1)
    best_checkpoint = glob.glob(os.path.join(out_dir, 'best_checkpoint.txt'))
    self.assertLen(best_checkpoint, 1)


if __name__ == '__main__':
  absltest.main()
