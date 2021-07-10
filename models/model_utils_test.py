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
