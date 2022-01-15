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
"""Tests for deepconsensus.models.networks."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import ml_collections
import numpy as np
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_configs
from deepconsensus.models import model_utils


def get_input_example(params: ml_collections.ConfigDict,
                      inference: bool) -> np.ndarray:
  """Returns one example from the training dataset for given params."""
  dataset = data_providers.get_dataset(
      file_pattern=params.train_path,
      num_epochs=params.num_epochs,
      batch_size=params.batch_size,
      params=params,
      inference=inference)
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
    inference = not training
    input_example = get_input_example(params, inference=inference)
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

  @parameterized.parameters(
      itertools.product(
          [
              'fc+test',
              'conv_net-resnet50+test',
              'transformer+test',
              'transformer_learn_values+test',
          ],
          [True, False]))
  def test_predict_and_model_fn_equal(self, config_name, inference):
    """Checks that model.predict and calling model as a function are equal."""
    config = model_configs.get_config(config_name)
    model_utils.modify_params(config)
    model = model_utils.get_model(config)
    input_example = get_input_example(config, inference=inference)
    softmax_output_predict = model.predict(input_example)
    softmax_output = model(input_example, training=False).numpy()
    self.assertTrue(
        np.allclose(softmax_output_predict, softmax_output, rtol=1e-05))

if __name__ == '__main__':
  absltest.main()
