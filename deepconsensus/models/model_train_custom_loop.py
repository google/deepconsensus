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
r"""Training binary for all neural network models using a custom training loop.

To use this binary for training a specific model, the corresponding config file
should be specified as input. Example usage:

CONFIG="//learning/genomics/deepconsensus/models/model_configs.py:transformer_learn_values+ccs"
OUT_DIR=/tmp

time blaze run -c opt \
//learning/genomics/deepconsensus/models:model_train_custom_loop -- \
  --params ${CONFIG} \
  --out_dir ${OUT_DIR} \
  --alsologtostderr
"""

import io
import json
import logging
import os
import random
from typing import Any, List, Optional, Tuple, Union

from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_utils


# pylint: disable=unused-import

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and model checkpoints.')
flags.DEFINE_string(
    'tpu', None, 'Name of the TPU to use. This gets '
    'populated automatically when using XManager.')
flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
flags.DEFINE_bool('debug', False,
                  'Enables dumping debug info for TensorBoard Debugger V2.')
flags.DEFINE_bool(
    'write_checkpoint_metrics', False,
    'Whether to write eval metrics for each checkpoint during training.')


class DTypeEncoder(json.JSONEncoder):
  """json encoder that allows for dtypes to be encoded."""

  def default(self, obj):
    if isinstance(obj, tf.DType):
      return repr(obj)
    else:
      return json.JSONEncoder.default(self, obj)


def save_params_as_json(out_dir: str,
                        params: ml_collections.ConfigDict) -> None:
  """Saves params to a JSON file."""
  json_path = os.path.join(out_dir, 'params.json')
  tf.io.gfile.makedirs(os.path.dirname(json_path))
  with tf.io.gfile.GFile(json_path, 'w') as json_file:
    json_file.write(json.dumps(dict(params), indent=4, cls=DTypeEncoder))


def get_datasets(
    params: ml_collections.ConfigDict, strategy: tf.distribute.Strategy
) -> Tuple[tf.distribute.DistributedDataset, tf.distribute.DistributedDataset]:
  """Returns datasets for training and evaluation."""
  train_input_fn = data_providers.create_input_fn(
      params=params, mode='train', limit=params.limit)
  eval_input_fn = data_providers.create_input_fn(
      params=params, mode='eval', limit=params.limit)
  train_dataset = strategy.experimental_distribute_dataset(train_input_fn())
  eval_dataset = strategy.experimental_distribute_dataset(eval_input_fn())
  return train_dataset, eval_dataset


def get_step_counts(params: ml_collections.ConfigDict) -> Tuple[int, int]:
  """Returns the steps for training and evaluation."""
  if params.limit <= 0:
    steps_per_epoch = params.n_examples_train // params.batch_size
    steps_per_eval = params.n_examples_eval // params.batch_size
  else:
    # When `params.limit` is set, use it to determine epoch size.
    steps_per_epoch = max(1, params.limit // params.batch_size)
    steps_per_eval = max(1, params.limit // params.batch_size)
  return steps_per_epoch, steps_per_eval


def get_checkpoint_and_initial_epoch(
    model: tf.keras.models.Model, optimizer: tf.keras.optimizers.Optimizer,
    out_dir: str, steps_per_epoch: int) -> Tuple[tf.train.Checkpoint, int]:
  """Loads a checkpoint if available and sets epoch to start training."""
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  latest_checkpoint = tf.train.latest_checkpoint(out_dir)
  initial_epoch = 0
  if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    logging.info('Loaded checkpoint %s', latest_checkpoint)
    initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
  return checkpoint, initial_epoch


def reset_all_metrics(metrics: List[tf.keras.metrics.Metric]) -> None:
  """Resets the values of provided metrics."""
  for metric in metrics:
    metric.reset_states()


def log_and_save_metrics(epoch: int, step: int, total_steps: int,
                         optimizer: tf.keras.optimizers.Optimizer,
                         loss_name: str, loss_value: float,
                         metrics: List[tf.keras.metrics.Metric],
                         training: bool) -> None:
  """Logs metrics and saves them for TensorBoard."""
  logging.info('epoch: %d  step: %d of %d loss: %f', epoch, step, total_steps,
               loss_value)
  if training:
    tf.summary.scalar('learning_rate', optimizer.lr, step=optimizer.iterations)
  tf.summary.scalar(loss_name, loss_value, step=optimizer.iterations)
  for metric in metrics:
    tf.summary.scalar(metric.name, metric.result(), step=optimizer.iterations)


def write_row(handle: Union[io.TextIOWrapper], row: List[Any]) -> None:
  """Formats an array as tab-delimited and writes."""
  handle.write('\t'.join(map(str, row)) + '\n')


def save_checkpoint(checkpoint: tf.train.Checkpoint, out_dir: str,
                    train_metrics: List[tf.keras.metrics.Metric],
                    eval_metrics: List[tf.keras.metrics.Metric],
                    write_checkpoint_metrics: bool) -> str:
  """Save checkpoint and return its name."""
  checkpoint_name = checkpoint.save(os.path.join(out_dir, 'checkpoint'))
  logging.info('Saved checkpoint to %s', checkpoint_name)
  logging.info('Logging checkpoint %s metrics.', checkpoint_name)
  metrics_file = os.path.join(out_dir, 'checkpoint_metrics.tsv')
  if write_checkpoint_metrics:
    if not tf.io.gfile.exists(metrics_file):
      with tf.io.gfile.GFile(metrics_file, 'w') as f:
        row = ['checkpoint_name', 'group', 'name', 'value']
        write_row(f, row)

    with tf.io.gfile.GFile(metrics_file, 'a') as f:
      for group_name, metrics in [('train', train_metrics),
                                  ('eval', eval_metrics)]:
        for metric in metrics:
          row = [
              checkpoint_name, group_name, metric.name,
              float(metric.result())
          ]
          write_row(f, row)
  return checkpoint_name


def train_model(out_dir: str, params: ml_collections.ConfigDict,
                strategy: tf.distribute.Strategy,
                write_checkpoint_metrics: bool) -> None:
  """Trains the model under the given strategy and params."""
  # Freeze config dict here to ensure it is hashable.
  params = ml_collections.FrozenConfigDict(params)
  save_params_as_json(out_dir, params)
  train_dataset, eval_dataset = get_datasets(params, strategy)
  steps_per_epoch, steps_per_eval = get_step_counts(params)

  with strategy.scope():
    logging.info('Building model.')
    model = model_utils.get_model(params)
    logging.info('Done building model.')
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metrics = model_utils.get_deepconsensus_metrics(name_prefix='train_')
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_metrics = model_utils.get_deepconsensus_metrics(name_prefix='eval_')
    loss_object = model_utils.get_deepconsensus_loss(
        params, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      # We divide per-replica losses by global batch size and sum this value
      # across all replicas to compute average loss scaled by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=params.batch_size)

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    checkpoint, initial_epoch = get_checkpoint_and_initial_epoch(
        model, optimizer, out_dir, steps_per_epoch)  # pytype: disable=wrong-arg-types  # typed-keras

  # Create summary writers
  train_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'train'))
  eval_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'eval'))

  def train_step(inputs):
    """Training StepFn."""
    features, labels = inputs
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = compute_loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    for metric in train_metrics:
      metric.update_state(labels, predictions)
    return loss

  def eval_step(inputs):
    """Eval StepFn."""
    features, labels = inputs
    predictions = model(features)
    loss = compute_loss(labels, predictions)
    eval_loss.update_state(loss)
    for metric in eval_metrics:
      metric.update_state(labels, predictions)
    return loss

  @tf.function
  def distributed_train_step(iterator):
    per_replica_losses = strategy.run(train_step, args=(next(iterator),))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  @tf.function
  def distributed_eval_step(iterator):
    per_replica_losses = strategy.run(eval_step, args=(next(iterator),))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  log_steps = 100
  train_iterator = iter(train_dataset)
  eval_iterator = iter(eval_dataset)
  min_eval_loss = 1e6
  for epoch in range(initial_epoch, params['num_epochs']):
    logging.info('Starting to run epoch: %s', epoch)
    with train_writer.as_default():
      for step in range(steps_per_epoch):
        reduced_train_loss = distributed_train_step(train_iterator)
        if step % log_steps == 0:
          log_and_save_metrics(
              epoch,
              step,
              steps_per_epoch,
              optimizer,
              train_loss.name,
              reduced_train_loss,
              train_metrics,
              training=True)
    with eval_writer.as_default():
      for step in range(steps_per_eval):
        reduced_eval_loss = distributed_eval_step(eval_iterator)
      log_and_save_metrics(
          epoch,
          step,
          steps_per_eval,
          optimizer,
          eval_loss.name,
          reduced_eval_loss,
          eval_metrics,
          training=False)
    checkpoint_name = save_checkpoint(checkpoint, out_dir, train_metrics,
                                      eval_metrics, write_checkpoint_metrics)
    if min_eval_loss > float(eval_loss.result()):
      min_eval_loss = float(eval_loss.result())
      with tf.io.gfile.GFile(os.path.join(out_dir, 'best_checkpoint.txt'),
                             'w') as f:
        f.write(os.path.basename(checkpoint_name))
    reset_all_metrics([train_loss, eval_loss] + train_metrics + eval_metrics)


def train(out_dir: str,
          params: ml_collections.ConfigDict,
          tpu: Optional[str],
          tpu_topology: Optional[str],
          write_checkpoint_metrics: bool,
          debug: Optional[bool] = False):
  """Run the model training and return evaluation output."""
  model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
  random.seed(params.seed)
  tf.random.set_seed(params.seed)
  os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'
  while True:
    try:
      if tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
      elif debug:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
      else:
        strategy = tf.distribute.MirroredStrategy()
      train_model(out_dir, params, strategy, write_checkpoint_metrics)
      break
    except tf.errors.UnavailableError:
      continue


def main(unused_args=None):
  train(FLAGS.out_dir, FLAGS.params, FLAGS.tpu, FLAGS.tpu_topology,
        FLAGS.write_checkpoint_metrics, FLAGS.debug)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'params',
      'out_dir',
  ])
  app.run(main)
