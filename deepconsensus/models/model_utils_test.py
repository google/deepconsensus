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
"""Tests for deepconsensus.models.model_utils."""

import os
import uuid

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from deepconsensus.models import model_configs
from deepconsensus.models import model_utils
from deepconsensus.utils import test_utils


class GetModelTest(absltest.TestCase):

  def test_valid_model_name(self):
    """Tests that correct model name works."""

    params = model_configs.get_config('fc+test')
    model_utils.modify_params(params)
    model = model_utils.get_model(params)
    self.assertIsInstance(model, tf.keras.Model)

  def test_invalid_model_name_throws_error(self):
    """Tests that incorrect model name throws an error."""

    with self.assertRaises(ValueError):
      params = model_configs.get_config('fc+test')
      model_utils.modify_params(params)
      params.model_name = 'incorrect_name'
      model_utils.get_model(params)


class ModifyParamsTest(parameterized.TestCase):

  @parameterized.parameters(['transformer+test', 'fc+test'])
  def test_params_modified(self, config_name):
    """Tests that params are correctly modified based on the model."""

    params = model_configs.get_config(config_name)

    # These params may have different values when running a sweep.
    # They should be modified so that they are equal.
    params.batch_size = 1
    params.default_batch_size = 2
    model_utils.modify_params(params)

    if config_name == 'fc+test':
      self.assertNotEqual(params.batch_size, params.default_batch_size)
    elif config_name == 'transformer+test':
      self.assertEqual(params.batch_size, params.default_batch_size)


class RunInferenceAndWriteResultsTest(absltest.TestCase):

  def test_output_dir_created(self):
    """Tests that output dir created when it does not exist."""

    out_dir = f'/tmp/output_dir/{uuid.uuid1()}'
    self.assertFalse(tf.io.gfile.isdir(out_dir))
    params = model_configs.get_config('transformer_learn_values+test')
    model_utils.modify_params(params)
    model = model_utils.get_model(params)
    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss=model_utils.get_deepconsensus_loss(params),
        metrics=model_utils.get_deepconsensus_metrics())
    model_utils.run_inference_and_write_results(model, out_dir, params)
    self.assertTrue(tf.io.gfile.isdir(out_dir))
    inference_output = os.path.join(out_dir, 'inference.csv')
    self.assertTrue(tf.io.gfile.exists(inference_output))
    with tf.io.gfile.GFile(inference_output) as inference_output_file:

      # Check that model.metric_names was called after real data was fed through
      # the model, and that metric names are correct.
      first_line = inference_output_file.readline()
      self.assertEqual(
          first_line, 'dataset,loss,' + ','.join([
              metric.name for metric in model_utils.get_deepconsensus_metrics()
          ]) + '\n')


if __name__ == '__main__':
  absltest.main()
