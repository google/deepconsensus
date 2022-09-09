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
"""Utilities for running training and inference."""

import io
import json
import logging
import os
from typing import Any, List, Optional, Tuple, Union

import ml_collections
import numpy as np
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import losses_and_metrics
from deepconsensus.models import model_configs
from deepconsensus.models import networks
from deepconsensus.models import transformer_basic_params



_YIELD_OVER_CSS_METRIC_NAME = 'yield_over_ccs'
_BATCH_IDENTITY_METRIC_NAME = 'per_batch_alignment_identity'


def get_deepconsensus_loss(
    params: ml_collections.ConfigDict,
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO
) -> tf.keras.losses.Loss:
  return {
      'xentropy':
          tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction),
      'alignment_loss':
          losses_and_metrics.AlignmentLoss(
              del_cost=params.del_cost,
              loss_reg=params.loss_reg,
              width=params.band_width,
              reduction=reduction),
  }[params.loss_function]


def get_deepconsensus_metrics(name_prefix='') -> List[tf.keras.metrics.Metric]:
  """Returns the metrics to use for training and evaluation."""
  return [
      losses_and_metrics.PerExampleAccuracy(
          name=f'{name_prefix}per_example_accuracy'),
      tf.keras.metrics.Mean(name=f'{name_prefix}{_BATCH_IDENTITY_METRIC_NAME}'),
      losses_and_metrics.YieldOverCCSMetric(
          name=f'{name_prefix}{_YIELD_OVER_CSS_METRIC_NAME}')
  ]


def update_metrics(metrics: List[tf.keras.metrics.Metric], labels: tf.Tensor,
                   predictions: tf.Tensor, identity_pred: tf.Tensor,
                   identity_ccs: tf.Tensor) -> None:
  """Updates metrics."""
  for metric in metrics:
    if _YIELD_OVER_CSS_METRIC_NAME in metric.name:
      metric.update_state(identity_ccs, identity_pred)
    elif _BATCH_IDENTITY_METRIC_NAME in metric.name:
      metric.update_state(identity_pred)
    else:
      metric.update_state(labels, predictions)


def get_record_shape(dataset_path: str) -> List[int]:
  """Returns an array that represents the shape of records in the given path.

  Input `dataset_path` should look something like
  /path/to/data/train/train, where the actual TFRecords are named something
  like /path/to/data/train/train-00228-of-00724.tfrecords.gz

  Args:
    dataset_path: string representing the sharded path for TFRecords. These
      records should be zipped tf.Examples protos with a subreads/shape field.
      This field has three values representing [hidden_size, max_length,
      channels].

  Raises:
    Exception: If no tfrecord files are found.
  """
  tfrecord_files = data_providers.create_glob_list(dataset_path)
  records = tf.data.TFRecordDataset(
      tfrecord_files, compression_type='GZIP').as_numpy_iterator()
  features = data_providers.parse_example(next(records))
  return list(map(int, features['subreads/shape'].numpy()))


def extract_example_height(dataset_sharded_path: str) -> int:
  """Gets example height based on a single entry within a dataset."""
  return get_record_shape(dataset_sharded_path)[0]


def get_ccs_from_example(features: tf.Tensor,
                         params: ml_collections.ConfigDict) -> tf.Tensor:
  """Gets CCS sequence from model input features."""
  _, _, _, _, ccs_index, _ = data_providers.get_indices(params['max_passes'])
  # CCS tensor with shape [batch_size, 1, max_length, 1].
  ccs = tf.gather(features, tf.range(*ccs_index), axis=1)
  # Return CCS tensor with shape [batch_size, max_length].
  return tf.squeeze(ccs, axis=[1, -1])


def get_model(params: ml_collections.ConfigDict) -> tf.keras.Model:
  """Returns desired model based on the given params."""
  if params.model_name == 'fc':
    model = networks.FullyConnectedNet(params)
  elif params.model_name == 'transformer':
    model = networks.EncoderOnlyTransformer(params)
  elif 'transformer_learn_values' in params.model_name:
    model = networks.EncoderOnlyLearnedValuesTransformer(params)
  else:
    raise ValueError('Unknown model name: %s' % params.model_name)
  return model




def del_param(params, name):
  if name in params:
    del params[name]


def modify_params(params: ml_collections.ConfigDict,
                  tpu: Optional[str] = None,
                  tpu_topology: Optional[str] = None,
                  speedy: bool = False,
                  max_length: Optional[int] = None,
                  is_training: bool = True) -> None:
  """Updates params as needed for the model and hardware being usued.

  This function should be called before working with a ConfigDict to ensure that
  derived configs that depend on other configs and hardware are correctly set
  before running any other code.

  Args:
    params: Config dictionary of the parameters to use.
    tpu: Name of the TPU being used or None.
    tpu_topology: of the form NxM, where N * M is the number of chips.
    speedy: Bool. Skip time-consuming steps that only add nice-to-have
        information.
    max_length: Equivalent to max_length in preprocess. If given, use this to
        set params.max_length instead of inspecting the examples.
    is_training: When not in training mode, do not run set_dataset

  Returns:
    Given config dictionary with some added or modified values based on the
    model and hardware used.
  """
  # Cannot set params that did not previously exist without unlocking.
  with params.unlocked():
    if not is_training:
      # Only allow dataset specification in params when in training mode.
      del_param(params, 'tf_dataset')
      del_param(params, 'train_path')
      del_param(params, 'eval_path')
      del_param(params, 'test_path')
      del_param(params, 'inference_path')
    # For all models, scale batch size by number of GPUs when using GPUs. If no
    # GPUs are being used, keep the original batch size.
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    if num_gpus > 0:
      assert not tpu
      logging.info('%d GPUs being used.', num_gpus)
      logging.info('Per-replica batch-size is %d.', params.batch_size)
      params.batch_size *= num_gpus
      logging.info('Global batch-size is %d.', params.batch_size)

    elif tpu is not None:
      # Add additional scale factor for TPU over the base batch size.
      params.batch_size *= params.tpu_scale_factor
      logging.info('Per-replica batch-size is %d.', params.batch_size)

      # We assume topology is in the format NxM. N * M represents the number of
      # chips, and there are two cores per chip.
      assert tpu_topology is not None
      tpu_topology_parts = list(map(int, tpu_topology.split('x')))
      assert len(tpu_topology_parts) == 2
      num_cores_used = tpu_topology_parts[0] * tpu_topology_parts[1] * 2
      params.batch_size *= num_cores_used
      logging.info('Global batch size is %d', params.batch_size)

    # Get max_length from dataset passed in, which is done at inference time.
    # Otherwise, we can expect params.train_path to exist.
    if max_length is not None:
      params.max_length = max_length

    if not hasattr(params, 'max_length'):
      raise ValueError('No params.max_length provided.')

    if 'transformer_learn_values' in params.model_name:
      dim = ((params.use_bases * params.per_base_hidden_size) +
             (params.use_pw * params.pw_hidden_size) +
             (params.use_ip * params.ip_hidden_size) +
             (params.use_strand * params.strand_hidden_size))
      params.hidden_size = ((params.max_passes * dim) +
                            (params.use_sn * params.sn_hidden_size * 4) +
                            (params.use_ccs * params.per_base_hidden_size))
    else:
      params.hidden_size = data_providers.get_total_rows(params.max_passes)

    if 'transformer' in params.model_name and params.hidden_size % 2 != 0:
      params.hidden_size += 1

    # Set model-specific parameters
    if params.model_name == 'transformer':
      # Transformer code uses default_batch_size, whereas my code uses
      # batch_size, so make sure both are the same.
      params.default_batch_size = params.batch_size
    elif 'transformer_learn_values' in params.model_name:
      # Transformer code uses default_batch_size, whereas my code uses
      # batch_size, so make sure both are the same.
      params.default_batch_size = params.batch_size
      if params.condense_transformer_input:
        logging.info('Setting hidden size to transformer_input_size.')
        params.hidden_size = params.transformer_input_size
    if 'transformer' in params.model_name:
      transformer_params = get_transformer_model_params(
          params.transformer_model_size, num_gpus=num_gpus)
      # Only add hyperparameters that don't already exist.
      for param_name, param_value in transformer_params.items():
        if param_name not in params:
          params[param_name] = param_value


def get_transformer_model_params(param_set, num_gpus):
  """Gets predefined transformer model params."""
  params_map = {
      'tiny': transformer_basic_params.TINY_PARAMS,
      'base': transformer_basic_params.BASE_PARAMS,
      'big': transformer_basic_params.BIG_PARAMS,
  }
  if num_gpus > 1:
    if param_set == 'big':
      return transformer_basic_params.BIG_MULTI_GPU_PARAMS.copy()
    elif param_set == 'base':
      return transformer_basic_params.BASE_MULTI_GPU_PARAMS.copy()
    else:
      raise ValueError('Not valid params: param_set={} num_gpus={}'.format(
          param_set, num_gpus))

  return params_map[param_set].copy()


def run_inference_and_write_results(model: tf.keras.Model,
                                    out_dir: str,
                                    params: ml_collections.ConfigDict,
                                    limit: int = -1):
  """Runs inference with given model and dataset and writes out results."""

  eval_paths = [params.eval_path]

  if not tf.io.gfile.isdir(out_dir):
    tf.io.gfile.makedirs(out_dir)
  logs_path = os.path.join(out_dir, 'inference.csv')
  with tf.io.gfile.GFile(logs_path, 'w') as logs_file:

    lines_to_write = []
    for path in eval_paths:
      validation_dataset = data_providers.get_dataset(
          file_pattern=path,
          # We only want to run one pass over the dataset.
          num_epochs=1,
          batch_size=params.batch_size,
          params=params,
          limit=limit,
          drop_remainder=False,
          # `inference` is set to False because this function is only used with
          # tf.Examples formatted for training mode as they contain a label.
          inference=False,
          example_label_tuple=True)

      history = model.evaluate(
          x=validation_dataset, batch_size=params.batch_size, steps=None)

      metric_values = ','.join(map(str, history))
      lines_to_write.append(f'{path},{metric_values}\n')

    # model.metric_names has to be referenced after model.evaluate(), otherwise
    # it is an empty string.
    metric_names = ','.join(model.metrics_names)
    lines_to_write = [f'dataset,{metric_names}\n'] + lines_to_write
    logs_file.write(''.join(lines_to_write))
    logs_file.write('\n')


def print_model_summary(model: tf.keras.Model, input_shape: Tuple[int, int, int,
                                                                  int]) -> None:
  """Runs a forward pass with dummy data then prints the model summary."""
  # Without calling this forward pass, we won't be able to print the summary.
  dummy_data = np.zeros(input_shape)
  _ = model(dummy_data)
  model.summary()


def read_params_from_json(checkpoint_path: str) -> ml_collections.ConfigDict:
  """Reads the params read from the params.json file for given checkpoint."""
  # For SavedModel, the path could be to the directory itself.
  param_set = model_configs.get_config()
  if os.path.isdir(checkpoint_path):
    dir_path = checkpoint_path
  else:
    dir_path = os.path.dirname(checkpoint_path)
  json_path = os.path.join(dir_path, 'params.json')
  json_params = ml_collections.ConfigDict(
      json.load(tf.io.gfile.GFile(json_path, 'r')))
  # Report new base parameters that are not present in params.json
  for b_param in param_set:
    if b_param not in json_params:
      logging.warning(('A new parameter (%s=%s) was added to the base config '
                       'that is not present in params.json'), b_param,
                      param_set[b_param])
  param_set.update(json_params)
  return param_set


def save_params_as_json(out_dir: str,
                        params: ml_collections.ConfigDict) -> None:
  """Saves params to a JSON file."""
  json_path = os.path.join(out_dir, 'params.json')
  tf.io.gfile.makedirs(os.path.dirname(json_path))
  with tf.io.gfile.GFile(json_path, 'w') as json_file:
    json_file.write(json.dumps(dict(params), indent=4))


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


def get_step_counts(params: ml_collections.ConfigDict,
                    eval_and_log_every_step: bool) -> Tuple[int, int]:
  """Returns the steps for training and evaluation."""

  if eval_and_log_every_step:
    steps_per_epoch = 1
    steps_per_eval = 1
  elif params.limit <= 0:
    steps_per_epoch = params.n_examples_train // params.batch_size
    steps_per_eval = params.n_examples_eval // params.batch_size
  else:
    # When `params.limit` is set, use it to determine epoch size.
    steps_per_epoch = max(1, params.limit // params.batch_size)
    steps_per_eval = max(1, params.limit // params.batch_size)
  return steps_per_epoch, steps_per_eval


def get_checkpoint_and_initial_epoch(
    model: tf.keras.models.Model, optimizer: tf.keras.optimizers.Optimizer,
    epoch_checkpoint: str) -> Tuple[tf.train.Checkpoint, int]:
  """Loads a checkpoint if available and sets epoch to start training."""
  initial_epoch = 0
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  if tf.io.gfile.exists(epoch_checkpoint):
    with tf.io.gfile.GFile(epoch_checkpoint, 'r') as f:
      epoch_checkpoint, initial_epoch = f.readline().split('\t')
      initial_epoch = int(initial_epoch)
      checkpoint.restore(epoch_checkpoint)
      logging.info('Loading checkpoint %s for epoch %s', epoch_checkpoint,
                   initial_epoch)
  else:
    logging.info('No Epoch checkpoint. Starting from epoch %s', initial_epoch)
    initial_epoch = 0
  return checkpoint, initial_epoch


def reset_all_metrics(metrics: List[tf.keras.metrics.Metric]) -> None:
  """Resets the values of provided metrics."""
  for metric in metrics:
    metric.reset_states()


def log_and_save_metrics(epoch: int, num_epochs: int, step: int,
                         total_steps: int,
                         optimizer: tf.keras.optimizers.Optimizer,
                         metrics: List[tf.keras.metrics.Metric], training: bool,
                         steps_per_second: float) -> None:
  """Logs metrics and saves them for TensorBoard."""
  logging.info(
      'epoch: %d  step: %d of %d metrics: %s', epoch, step, total_steps,
      ' '.join(f'{metric.name}= {metric.result()}' for metric in metrics))

  overall_progress = optimizer.iterations.numpy() / (total_steps * num_epochs)


  if training:
    tf.summary.scalar('learning_rate', optimizer.lr, step=optimizer.iterations)
    tf.summary.scalar('progress/epoch', epoch, step=optimizer.iterations)
    tf.summary.scalar(
        'progress/overall_progress',
        overall_progress,
        step=optimizer.iterations)
  for metric in metrics:
    tf.summary.scalar(metric.name, metric.result(), step=optimizer.iterations)
    metric.reset_states()


def write_row(handle: Union[io.TextIOWrapper], row: List[Any]) -> None:
  """Formats an array as tab-delimited and writes."""
  handle.write('\t'.join(map(str, row)) + '\n')


def save_checkpoint(checkpoint: tf.train.Checkpoint, out_dir: str,
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
      for group_name, metrics in [('eval', eval_metrics)]:
        for metric in metrics:
          row = [
              checkpoint_name, group_name, metric.name,
              float(metric.result())
          ]
          write_row(f, row)
  return checkpoint_name
