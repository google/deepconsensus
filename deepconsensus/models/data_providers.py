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
"""Functions for yielding input arrays for models."""

import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union

import ml_collections
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf

from deepconsensus.utils import dc_constants


# Define base fields for TFRecords.
PROTO_FEATURES_INFERENCE = {
    'name': tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    'window_pos': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'subreads/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    # Shapes are written to the int64_list of the example.
    'subreads/shape': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    'subreads/num_passes': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
}
# Add on label fields to train proto.
PROTO_FEATURES_TRAIN = dict(
    {
        'label/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'label/shape': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    }, **PROTO_FEATURES_INFERENCE)


def get_total_rows(max_passes: int) -> int:
  """Returns total rows in input tf.Examples. Update if other signals added."""
  # For each of `max_subreads`, we have three pieces of information: bases, PW,
  # and IP. We also have four rows for SN, and one for strand.
  # The information is structured as follows:
  # Bases: (0, params.max_passes - 1) represent bases.
  # PW: rows params.max_passes to (params.max_passes * 2 - 1)
  # IP: rows (params.max_passes * 2) to (params.max_passes * 3 - 1)
  # Strand: rows (params.max_passes * 3) to (params.max_passes * 4)
  # CCS+SN: rows (params.max_passes * 4 + 1) to (params.max_passes * 4 + 5)
  # The last five rows are CCS sequence (1), and SN (4).
  return (max_passes * 4) + 5


def get_indices(max_passes: int) -> Iterable[Tuple[int, int]]:
  """Return row indices for bases/PW/IP/SN in tf.Example subreads array."""
  base_indices = (0, max_passes)
  pw_indices = (max_passes, max_passes * 2)
  ip_indices = (max_passes * 2, max_passes * 3)
  strand_indices = (max_passes * 3, max_passes * 4)
  ccs_indices = (max_passes * 4, max_passes * 4 + 1)
  sn_indices = (max_passes * 4 + 1, max_passes * 4 + 5)
  return base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices


@tf.function
def remove_internal_gaps_and_shift(label: tf.Tensor) -> tf.Tensor:
  """Filters internal gaps and shifts sequences to the left."""
  label = tf.squeeze(label)
  subset = tf.transpose(
      tf.gather(label, tf.where(label != dc_constants.GAP_OR_PAD_INT)))
  pad_amt = tf.shape(label)[0] - tf.shape(subset)[1]
  padded = tf.pad(subset, [[0, 0], [0, pad_amt]])
  return tf.squeeze(padded)


def format_rows(
    subreads: tf.Tensor,
    params: ml_collections.FrozenConfigDict,
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True,
) -> tf.Tensor:
  """Returns model input matrix formatted based on input args."""
  base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices = get_indices(
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
    num_rows = get_total_rows(params.max_passes)

  rows.set_shape((num_rows, params.max_length, params.num_channels))
  return rows


def process_feature_dict(
    features: Dict[str, tf.Tensor],
    params: Union[config_dict.ConfigDict, config_dict.FrozenConfigDict],
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Parses a serialized tf.Example to return an input, label, and metadata.

  Args:
    features: Dictionary of features to process for the model.
    params: A config dictionary containing desired hyperparameters.
    cap_pw: If True, pulse width values are capped.
    cap_ip: If True, interpulse distance values are capped.
    cap_sn: If True, signal to noise ratio values are capped.

  Returns:
    rows: Input matrix that will be fed into neural networks for training.
    label: Label vector that will be used for training.
    num_passes: The number of subreads present in this example.
  """
  label = tf.convert_to_tensor(np.array([]))
  subreads = features['subreads']
  num_passes = features['subreads/num_passes']
  rows = format_rows(
      subreads=subreads,
      params=params,
      cap_pw=cap_pw,
      cap_ip=cap_ip,
      cap_sn=cap_sn)
  return rows, label, num_passes, features['window_pos'], features['name']


def parse_example(proto_string, inference=False):
  if inference:
    proto_features = PROTO_FEATURES_INFERENCE
  else:
    proto_features = PROTO_FEATURES_TRAIN
  parsed_features = tf.io.parse_single_example(
      serialized=proto_string, features=proto_features)
  return parsed_features


def process_input(
    proto_string: Union[tf.Tensor, bytes],
    params: ml_collections.FrozenConfigDict,
    inference: bool,
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Parses a serialized tf.Example to return an input, label, and metadata.

  Args:
    proto_string: A tensor containing the serialized tf.Example string.
    params: A config dictionary containing desired hyperparameters.
    inference: Whether to parse tf.Examples for inference or training.
    cap_pw: If True, pulse width values are capped.
    cap_ip: If True, interpulse distance values are capped.
    cap_sn: If True, signal to noise ratio values are capped.

  Returns:
    rows: Input matrix that will be fed into neural networks for training.
    label: Label vector that will be used for training.
    num_passes: The number of subreads present in this example.
  """
  parsed_features = parse_example(proto_string, inference)
  flat_subreads = tf.io.decode_raw(parsed_features['subreads/encoded'],
                                   dc_constants.TF_DATA_TYPE)
  subreads = tf.reshape(flat_subreads, parsed_features['subreads/shape'])
  num_passes = tf.cast(parsed_features['subreads/num_passes'],
                       dc_constants.TF_DATA_TYPE)
  if not inference:
    flat_label = tf.io.decode_raw(parsed_features['label/encoded'],
                                  dc_constants.TF_DATA_TYPE)
    label = tf.reshape(flat_label, parsed_features['label/shape'])

    if params.remove_label_gaps:
      label = remove_internal_gaps_and_shift(label)
    label.set_shape((params.max_length))
  else:
    label = tf.convert_to_tensor(np.array([]))
  rows = format_rows(
      subreads=subreads,
      params=params,
      cap_pw=cap_pw,
      cap_ip=cap_ip,
      cap_sn=cap_sn)
  return rows, label, num_passes


def drop_metadata(*args):
  """Return only subreads and label."""
  subreads = args[0]
  label = args[1]
  return (subreads, label)


def get_dataset(file_pattern: str,
                num_epochs: Optional[int],
                batch_size: int,
                params: Union[ml_collections.ConfigDict,
                              ml_collections.FrozenConfigDict],
                inference: bool,
                cap_pw: bool = True,
                cap_ip: bool = True,
                limit: int = -1,
                drop_remainder: bool = True,
                keep_metadata: bool = False) -> tf.data.Dataset:
  """Parses TFRecords and return a dataset."""

  # Output type annotations added for clarity, but Pytype only expects
  # Tuple[Any, Any] here and does not check for tf.Tensors.
  def _process_input_helper(
      proto_string: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return process_input(
        proto_string=proto_string,
        params=params,
        cap_pw=cap_pw,
        cap_ip=cap_ip,
        inference=inference)

  file_patterns = create_glob_list(file_pattern)
  ds = tf.data.TFRecordDataset(file_patterns, compression_type='GZIP')
  ds = ds.map(map_func=_process_input_helper)
  ds = ds.shuffle(buffer_size=params.buffer_size, reshuffle_each_iteration=True)
  if num_epochs:
    ds = ds.repeat(num_epochs)

  # When training, we can drop num_passes, window_pos, and name.
  if not keep_metadata:
    ds = ds.map(
        drop_metadata, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
  ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.take(limit)
  return ds


def create_glob_list(paths: Union[str, List[str]]) -> List[str]:
  """Creates a globbed file list."""
  file_patterns = []
  if isinstance(paths, str):
    paths = [paths]
  for path in paths:
    file_patterns.append(tf.io.gfile.glob(path))
  return list(itertools.chain(*file_patterns))


def create_input_fn(params, mode, limit: int = -1, drop_remainder: bool = True):
  """Returns an input function that will return a tfrecord based dataset."""

  def _process_input_helper(
      proto_string: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Set inference to False here because we only use this function with
    # tf.Examples that have labels present.
    return process_input(
        proto_string=proto_string,
        params=params,
        inference=False,
        cap_pw=True,
        cap_ip=True)

  def input_fn() -> tf.data.Dataset:
    """Prepares a dataset for training or evaluation."""
    is_training = (mode == 'train')
    batch_size = params.batch_size
    assert mode in ['train', 'eval']
    file_patterns = create_glob_list(params[f'{mode}_path'])
    ds = tf.data.Dataset.list_files(file_patterns, shuffle=is_training)
    ds = ds.repeat()
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    if is_training:
      ds = ds.shuffle(
          buffer_size=params['buffer_size'], reshuffle_each_iteration=True)

    # Best practices suggest batching first, but this map errors out when we
    # batch first, so do this before mapping.
    ds = ds.map(
        _process_input_helper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    # Best practices suggest batching before mapping.
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        drop_metadata, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = ds.take(limit)
    return ds

  return input_fn
