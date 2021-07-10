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
"""Functions for yielding input arrays for models."""

import os
from typing import Optional, Tuple, Union

import ml_collections
import tensorflow.compat.v2 as tf

from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.utils import dc_constants


# Define field names, types, and sizes for TFRecords.
_PROTO_FEATURES = {
    'subreads/encoded':
        tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    # Shapes are written to the int64_list of the example.
    'subreads/shape':
        tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    'subreads/num_passes':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'label/encoded':
        tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'label/shape':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'deepconsensus_input/encoded':
        tf.io.FixedLenFeature(shape=[], dtype=tf.string),
}


@tf.function
def remove_internal_gaps_and_shift(label: tf.Tensor) -> tf.Tensor:
  """Filters internal gaps and shifts sequences to the left."""
  label = tf.squeeze(label)
  subset = tf.transpose(
      tf.gather(label, tf.where(label != dc_constants.GAP_OR_PAD_INT)))
  pad_amt = tf.shape(label)[0] - tf.shape(subset)[1]
  padded = tf.pad(subset, [[0, 0], [0, pad_amt]])
  return tf.squeeze(padded)


def process_input(
    proto_string: Union[tf.Tensor, bytes],
    params: ml_collections.FrozenConfigDict,
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Parses a serialized tf.Example to return an input, label, and metadata.

  Args:
    proto_string: A tensor containing the serialized tf.Example string.
    params: A config dictionary containing desired hyperparameters.
    cap_pw: If True, pulse width values are capped.
    cap_ip: If True, interpulse distance values are capped.
    cap_sn: If True, signal to noise ratio values are capped.

  Returns:
    rows: Input matrix that will be fed into neural networks for training.
    label: Label vector that will be used for training.
    num_passes: The number of subreads present in this example.
    encoded_deepconsensus_input: A tensor containing the serialized
      DeepConsensusInput proto for this example.
  """
  parsed_features = tf.io.parse_single_example(
      serialized=proto_string, features=_PROTO_FEATURES)
  flat_subreads = tf.io.decode_raw(parsed_features['subreads/encoded'],
                                   dc_constants.TF_DATA_TYPE)
  subreads = tf.reshape(flat_subreads, parsed_features['subreads/shape'])
  num_passes = tf.cast(parsed_features['subreads/num_passes'],
                       dc_constants.TF_DATA_TYPE)
  flat_label = tf.io.decode_raw(parsed_features['label/encoded'],
                                dc_constants.TF_DATA_TYPE)
  label = tf.reshape(flat_label, parsed_features['label/shape'])

  if params.remove_label_gaps:
    label = remove_internal_gaps_and_shift(label)

  encoded_deepconsensus_input = parsed_features['deepconsensus_input/encoded']
  base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices = tf_example_utils.get_indices(
      params.max_passes)
  base_rows = subreads[slice(*base_indices)]
  pw_rows = subreads[slice(*pw_indices)]
  ip_rows = subreads[slice(*ip_indices)]
  strand_rows = subreads[slice(*strand_indices)]
  ccs_rows = subreads[slice(*ccs_indices)]
  sn_rows = subreads[slice(*sn_indices)]

  if cap_pw:
    pw_rows = tf.clip_by_value(
        pw_rows, clip_value_min=0, clip_value_max=dc_constants.PW_MAX)
  if cap_ip:
    ip_rows = tf.clip_by_value(
        ip_rows, clip_value_min=0, clip_value_max=dc_constants.IP_MAX)
  if cap_sn:
    sn_rows = tf.clip_by_value(
        sn_rows, clip_value_min=0, clip_value_max=dc_constants.SN_MAX)
  if params.get('input_format') == 'stack[base,pw,ip,sn]':
    ccs_rows = tf.image.pad_to_bounding_box(ccs_rows, 0, 0, params.max_passes,
                                            params.max_length)
    sn_rows = tf.image.pad_to_bounding_box(sn_rows, 0, 0, params.max_passes,
                                           params.max_length)
    rows = tf.concat([base_rows, pw_rows, ip_rows, ccs_rows, sn_rows], axis=-1)
    if params.max_passes < 32:
      rows = tf.image.pad_to_bounding_box(rows, 0, 0, 32, params.max_length)
    num_rows = max(params.max_passes, 32)
  else:
    rows = tf.concat(
        [base_rows, pw_rows, ip_rows, strand_rows, ccs_rows, sn_rows], axis=0)
    num_rows = tf_example_utils.get_total_rows(params.max_passes)
  label.set_shape((params.max_length))
  rows.set_shape((num_rows, params.max_length, params.num_channels))
  return rows, label, num_passes, encoded_deepconsensus_input


def get_dataset_with_metadata(file_pattern: str,
                              num_epochs: Optional[int],
                              batch_size: int,
                              params: ml_collections.FrozenConfigDict,
                              cap_pw: bool = True,
                              cap_ip: bool = True,
                              limit: int = -1,
                              drop_remainder: bool = True) -> tf.data.Dataset:
  """Parses TFRecords and return a dataset with metadata."""

  # Output type annotations added for clarity, but Pytype only expects
  # Tuple[Any, Any] here and does not check for tf.Tensors.
  def _process_input_helper(
      proto_string: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return process_input(
        proto_string=proto_string, params=params, cap_pw=cap_pw, cap_ip=cap_ip)

  filenames = tf.io.gfile.glob(file_pattern)
  ds = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
  ds = ds.map(map_func=_process_input_helper)
  ds = ds.shuffle(buffer_size=params.buffer_size, reshuffle_each_iteration=True)
  ds = ds.repeat(num_epochs)
  ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.take(limit)
  return ds


def get_dataset(file_pattern: str,
                num_epochs: Optional[int],
                batch_size: int,
                params: Union[ml_collections.ConfigDict,
                              ml_collections.FrozenConfigDict],
                cap_pw: bool = True,
                cap_ip: bool = True,
                limit: int = -1,
                drop_remainder: bool = True) -> tf.data.Dataset:
  """Parses TFRecords and return a dataset."""

  ds = get_dataset_with_metadata(
      file_pattern=file_pattern,
      num_epochs=num_epochs,
      batch_size=batch_size,
      params=params,
      cap_pw=cap_pw,
      cap_ip=cap_ip,
      limit=limit,
      drop_remainder=drop_remainder)

  def _remove_metadata(subreads, label, num_passes, deepconsensus_input):
    """Returns only subread and label, without keeping metadata."""
    del num_passes, deepconsensus_input
    return subreads, label

  ds = ds.map(_remove_metadata)
  return ds


# <internal>
# approach for training.
def create_input_fn(params, mode, limit: int = -1, drop_remainder: bool = True):
  """Returns an input function that will return a tfrecord based dataset."""

  def _process_input_helper(
      proto_string: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return process_input(
        proto_string=proto_string, params=params, cap_pw=True, cap_ip=True)

  def input_fn() -> tf.data.Dataset:
    """Prepares a dataset for training or evaluation."""
    is_training = (mode == 'train')
    batch_size = params.batch_size
    if mode == 'train':
      file_pattern = os.path.join(params['train_path'], '*')
    else:
      file_pattern = os.path.join(params['eval_path'], '*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    if is_training:
      dataset = dataset.shuffle(
          buffer_size=params['buffer_size'], reshuffle_each_iteration=True)

    # Best practices suggest batching first, but this map errors out when we
    # batch first, so do this before mapping.
    dataset = dataset.map(
        _process_input_helper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    # Best practices suggest batching before mapping.
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(
        lambda subreads, label, num_passes, dc_input: (subreads, label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.take(limit)
    return dataset

  return input_fn
