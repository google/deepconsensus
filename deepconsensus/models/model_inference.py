r"""Inference binary for all neural network models.

To use this binary for running inference with a specific model, the
corresponding config does not need to be specified and will be inferred. Example
usage:

OUT_DIR=/tmp
CHECKPOINT_PATH=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/experiments/20201125/exp_18828140/wu_2/model
time blaze run -c opt \
//learning/genomics/deepconsensus/models:model_inference -- \
  --out_dir ${OUT_DIR} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --alsologtostderr
"""

import random
from typing import Optional

from absl import app
from absl import flags
import ml_collections
import tensorflow as tf

from deepconsensus.models import model_utils

FLAGS = flags.FLAGS
ml_collections.config_flags.DEFINE_config_file('params', None,
                                               'Training configuration.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to checkpoint that will be loaded in.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and model predictions.')
flags.DEFINE_string(
    'master', None, 'Name of the TPU to use. This gets '
    'populated automatically when using XManager.')
flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
flags.DEFINE_integer(
    'limit', -1, 'Limit to N records per train/tune dataset. '
    '-1 will evaluate all examples.')


def run_inference(out_dir: str, params: ml_collections.ConfigDict,
                  checkpoint_path: str, master: Optional[str],
                  tpu_topology: Optional[str], limit: int):
  """Runs model evaluation with an existing checkpoint."""
  model_utils.modify_params(params, tpu=master, tpu_topology=tpu_topology)

  # Set seed for reproducibility.
  random.seed(params.seed)
  tf.random.set_seed(params.seed)

  # <internal>
  # may need to explicitly distribute data to the workers to see speedup.
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = model_utils.get_model(params)
    try:
      model.load_weights(checkpoint_path)
    except AssertionError:
      # Use this approach for models saved in tf.train.Checkpoint format through
      # the custom training loop code.
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
    params = model_utils.read_params_from_json(
        checkpoint_path=FLAGS.checkpoint_path)
  else:
    params = FLAGS.params
  run_inference(FLAGS.out_dir, params, FLAGS.checkpoint_path, FLAGS.master,
                FLAGS.tpu_topology, FLAGS.limit)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'out_dir',
      'checkpoint_path',
  ])
  app.run(main)
