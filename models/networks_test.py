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
"""Tests for deepconsensus.models.networks."""

import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized

import ml_collections
import numpy as np
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_configs
from deepconsensus.models import model_utils


def get_input_example(params: ml_collections.ConfigDict) -> np.ndarray:
  """Returns one example from the training dataset for given params."""
  dataset = data_providers.get_dataset(
      file_pattern=os.path.join(params.train_path, '*'),
      num_epochs=params.num_epochs,
      batch_size=params.batch_size,
      params=params)
  input_example, _ = next(dataset.as_numpy_iterator())
  return input_example


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          [True, False],
          [
              'fc+test',
              'conv_net-resnet50+test',
              'transformer+test',
              'transformer_learn_values+test',
          ],
          [True, False]))
  def test_outputs(self, training, config_name, use_predict):
    """Checks that softmax distribution and final predictions are valid.

    This test is only checking the output format and does not train the model.
    Args:
      training: whether we are in training or eval/test mode.
      config_name: config to test.
      use_predict: whether to use model.predict or call model as a function.
    """
    params = model_configs.get_config(config_name)
    model_utils.modify_params(params)
    model = model_utils.get_model(params)
    input_example = get_input_example(params)
    if use_predict:
      softmax_output = model.predict(input_example)
    else:
      softmax_output = model(input_example, training=training).numpy()
    predictions = tf.argmax(softmax_output, -1)

    # First dimension will always be equal to batch_size because test config
    # uses a batch size of 1.
    self.assertEqual(softmax_output.shape,
                     (params.batch_size, params.max_length, params.num_classes))
    self.assertTrue(
        np.allclose(
            np.sum(softmax_output, axis=-1),
            np.ones(shape=[params.batch_size, params.max_length])))
    self.assertEqual(predictions.shape, (params.batch_size, params.max_length))

  @parameterized.parameters([
      'fc+test',
      'conv_net-resnet50+test',
      'transformer+test',
      'transformer_learn_values+test',
  ])
  def test_predict_and_model_fn_equal(self, config_name):
    """Checks that model.predict and calling model as a function are equal."""
    config = model_configs.get_config(config_name)
    model_utils.modify_params(config)
    model = model_utils.get_model(config)
    input_example = get_input_example(config)
    softmax_output_predict = model.predict(input_example)
    softmax_output = model(input_example, training=False).numpy()
    self.assertTrue(np.array_equal(softmax_output_predict, softmax_output))


if __name__ == '__main__':
  absltest.main()
