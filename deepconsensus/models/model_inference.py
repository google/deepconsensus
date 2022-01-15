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
r"""Inference binary for all neural network models.

To use this binary for running inference with a specific model, the
corresponding config does not need to be specified and will be inferred. Example
usage:

OUT_DIR=/tmp
CHECKPOINT=<internal>
time blaze run -c opt \
//learning/genomics/deepconsensus/models:model_inference -- \
  --out_dir ${OUT_DIR} \
  --checkpoint ${CHECKPOINT} \
  --alsologtostderr
"""

import random
from typing import Optional

from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

from deepconsensus.models import model_utils

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
flags.DEFINE_string('checkpoint', None,
                    'Path to checkpoint that will be loaded in.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and model predictions.')
flags.DEFINE_string(
    'tpu', None, 'Name of the TPU to use. This gets '
    'populated automatically when using XManager.')
flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
flags.DEFINE_integer(
    'limit', -1, 'Limit to N records per train/tune dataset. '
    '-1 will evaluate all examples.')


def run_inference(out_dir: str, params: ml_collections.ConfigDict,
                  checkpoint_path: str, tpu: Optional[str],
                  tpu_topology: Optional[str], limit: int):
  """Runs model evaluation with an existing checkpoint."""
  model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)

  # Set seed for reproducibility.
  random.seed(params.seed)
  tf.random.set_seed(params.seed)

  # TODO: multiple GPUs don't provide a speedup for inference. We
  # may need to explicitly distribute data to the workers to see speedup.
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = model_utils.get_model(params)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss=model_utils.get_deepconsensus_loss(params),
        metrics=model_utils.get_deepconsensus_metrics())

    input_shape = (1, params.hidden_size, params.max_length,
                   params.num_channels)
    model_utils.print_model_summary(model, input_shape)

    model_utils.run_inference_and_write_results(
        model=model, out_dir=out_dir, params=params, limit=limit)


def main(unused_args=None):
  if not FLAGS.params:
    params = model_utils.read_params_from_json(checkpoint_path=FLAGS.checkpoint)
  else:
    params = FLAGS.params
  run_inference(FLAGS.out_dir, params, FLAGS.checkpoint, FLAGS.tpu,
                FLAGS.tpu_topology, FLAGS.limit)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'out_dir',
      'checkpoint',
  ])
  app.run(main)
