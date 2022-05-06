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
r"""Convert a Checkpoint to a SavedModel.

Example command:
  convert_to_saved_model --checkpoint=/path/to/checkpoint --output=/tmp/output
"""

import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_utils
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

# Outputs:
flags.DEFINE_string('output', None, 'Output SavedModel name.')

# Model checkpoint:
flags.DEFINE_string(
    'checkpoint',
    None, 'Path to checkpoint directory + prefix. '
    'For example: <path/to/model>/checkpoint-50.',
    required=True)


def register_required_flags():
  flags.mark_flags_as_required([
      'checkpoint',
      'output',
  ])


def initialize_model(checkpoint_path: str) -> Optional[tf.keras.Model]:
  """Initializes the model and gathers parameters.

  Args:
    checkpoint_path: Path to model checkpoint.

  Returns:
    An initialized model.
  """
  params = model_utils.read_params_from_json(checkpoint_path=checkpoint_path)
  logging.info('Loading %s', checkpoint_path)
  model = model_utils.get_model(params)
  # This loads a model saved in tf.train.Checkpoint format through the custom
  # training loop code.
  checkpoint = tf.train.Checkpoint(model=model)
  # Note that the `print_model_summary` is necessary because we need to run a
  # forward pass with the model in order for assert_existing_objects_matched to
  # work as expected. If you don't do this, then assert_existing_objects_matched
  # will not raise an error even if the wrong checkpoint is used.
  # Some context here: b/148023980.
  row_size = data_providers.get_total_rows(params.max_passes)
  input_shape = (1, row_size, params.max_length, params.num_channels)
  model_utils.print_model_summary(model, input_shape)
  checkpoint.restore(
      checkpoint_path).expect_partial().assert_existing_objects_matched()

  logging.info('Finished initialize_model.')
  return model


def main(_):
  """Main entry point."""
  loaded_model = initialize_model(checkpoint_path=FLAGS.checkpoint)
  tf.saved_model.save(loaded_model, FLAGS.output)
  # Copy over the params.json. At this point, we know params.json exists.
  json_path = os.path.join(os.path.dirname(FLAGS.checkpoint), 'params.json')
  gfile.Copy(
      json_path, os.path.join(FLAGS.output, 'params.json'), overwrite=True)


if __name__ == '__main__':
  register_required_flags()
  app.run(main)
