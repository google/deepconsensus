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

import json
import logging
import os
from typing import List, Optional, Tuple

import ml_collections
import numpy as np
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import losses_and_metrics
from deepconsensus.models import networks
from deepconsensus.utils import dc_constants
from official.nlp.transformer import misc


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
              reduction=reduction),
  }[params.loss_function]


def get_deepconsensus_metrics(name_prefix='') -> List[tf.keras.metrics.Metric]:
  """Returns the metrics to use for training and evaluation."""
  class_to_name = {
      base: dc_constants.VOCAB.index(base) for base in dc_constants.VOCAB
  }
  class_to_name['gap_or_pad'] = class_to_name[dc_constants.GAP_OR_PAD]
  del class_to_name[dc_constants.GAP_OR_PAD]
  per_class_accuracy_metrics = [
      losses_and_metrics.PerClassAccuracy(
          class_value=class_value, name=f'{name_prefix}{name}')
      for name, class_value in class_to_name.items()
  ]
  return [
      tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name_prefix}accuracy'),
      losses_and_metrics.PerExampleAccuracy(
          name=f'{name_prefix}per_example_accuracy'),
  ] + per_class_accuracy_metrics


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


def extract_max_length(dataset_sharded_path: str) -> int:
  """Extracts the example length based on a single entry within a dataset."""
  return get_record_shape(dataset_sharded_path)[1]


def extract_example_height(dataset_sharded_path: str) -> int:
  """Gets example height based on a single entry within a dataset."""
  return get_record_shape(dataset_sharded_path)[0]


def get_model(params: ml_collections.ConfigDict) -> tf.keras.Model:
  """Returns desired model based on the given params."""
  if params.model_name == 'fc':
    model = networks.FullyConnectedNet(params)
  elif params.model_name == 'conv_net':
    model = networks.ConvNet(params)
  elif params.model_name == 'transformer':
    model = networks.EncoderOnlyTransformer(params)
  elif params.model_name == 'transformer_learn_values':
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
                  dataset_path: Optional[str] = None,
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
    dataset_path: Optional. Path to dataset from which to extract max_length.
    speedy: Bool. Skip time-consuming steps that only add nice-to-have
        information.
    max_length: Equivalent to padded_len in preprocess. If given, use this to
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
    elif dataset_path:
      params.max_length = extract_max_length(
          os.path.join(dataset_path, os.path.basename(dataset_path)))
    else:
      params.max_length = extract_max_length(params.train_path)

    if params.model_name == 'transformer_learn_values':
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
    if params.model_name == 'conv_net':
      # If max_passes < 32; image will be padded as to a height of(32) req'd.
      params.hidden_size = max(32, params.max_passes)
    elif params.model_name == 'transformer':
      # Transformer code uses default_batch_size, whereas my code uses
      # batch_size, so make sure both are the same.
      params.default_batch_size = params.batch_size
    elif params.model_name == 'transformer_learn_values':
      # Transformer code uses default_batch_size, whereas my code uses
      # batch_size, so make sure both are the same.
      params.default_batch_size = params.batch_size
      if params.condense_transformer_input:
        params.hidden_size = params.transformer_input_size
    if 'transformer' in params.model_name:
      transformer_params = misc.get_model_params(
          params.transformer_model_size, num_gpus=num_gpus)
      # Only add hyperparameters that don't already exist.
      for param_name, param_value in transformer_params.items():
        if param_name not in params:
          params[param_name] = param_value


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
          inference=False)

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
  json_path = os.path.join(os.path.dirname(checkpoint_path), 'params.json')
  params = ml_collections.ConfigDict(
      json.load(tf.io.gfile.GFile(json_path, 'r')))
  # This cannot be passed in to be updated as json serializes it as a string.
  # This value shouldn't change across experiments, so just use what is present
  # in cuurrent config, rather than the experiment config.
  if params.dtype:
    del params.dtype
    params.dtype = dc_constants.TF_DATA_TYPE
  return params
