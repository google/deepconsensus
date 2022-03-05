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
    # We encountered the issue in
    # https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    # with MirroredStrategy. So, setting debug=True to avoid using
    # MirroredStrategy for testing.
    model_train_custom_loop.train(
        out_dir=out_dir,
        params=params,
        tpu=tpu,
        tpu_topology=tpu_topology,
        write_checkpoint_metrics=True,
        debug=True)

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


class GetStepCountsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple',
          n_examples_train=1000,
          n_examples_eval=100,
          batch_size=10,
          limit=-1,
          expected_step_counts=(100, 10)),
      dict(
          testcase_name='with_limit',
          n_examples_train=1000,
          n_examples_eval=100,
          batch_size=10,
          limit=100,
          expected_step_counts=(10, 10)),
  )
  def test_get_step_counts(self, n_examples_train, n_examples_eval, batch_size,
                           limit, expected_step_counts):
    params = model_configs.get_config('fc+test')
    with params.unlocked():
      params.n_examples_train = n_examples_train
      params.n_examples_eval = n_examples_eval
      params.limit = limit
      params.batch_size = batch_size

    self.assertEqual(
        model_train_custom_loop.get_step_counts(params), expected_step_counts)


if __name__ == '__main__':
  absltest.main()
