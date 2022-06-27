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
from deepconsensus.utils import utils


# Define base fields for TFRecords.
PROTO_FEATURES_INFERENCE = {
    'name': tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    'window_pos': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'subreads/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    # Shapes are written to the int64_list of the example.
    'subreads/shape': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
    'subreads/num_passes': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'ccs_base_quality_scores': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
}

# Add on label fields to train proto.
PROTO_FEATURES_TRAIN = dict(
    {
        'label/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'label/shape': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    }, **PROTO_FEATURES_INFERENCE)


def get_total_rows(max_passes: int) -> int:
  """Returns total rows in input tf.Examples.

  Update if other signals added.

  For each of `max_subreads`, we have four pieces of information: bases, PW, IP,
  and strand. We also have one row for CCS, and four rows for SN (in that
  order).
  The information is structured as follows:
  Bases: rows 0 to  (params.max_passes - 1)
  PW: rows (params.max_passes) to (params.max_passes * 2 - 1)
  IP: rows (params.max_passes * 2) to (params.max_passes * 3 - 1)
  Strand: rows (params.max_passes * 3) to (params.max_passes * 4 - 1)
  CCS+SN: rows (params.max_passes * 4) to (params.max_passes * 4 + 5)

  Args:
    max_passes: Maximum number of subreads to show. Space is made for them all
      even though few examples will have enough subreads to fill these rows.
  Returns: Total number of rows in the full example.
  """
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
    max_passes: int,
    max_length: int,
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True,
) -> tf.Tensor:
  """Returns model input matrix formatted based on input args."""
  base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices = get_indices(
      max_passes)
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
  rows = tf.concat(
      [base_rows, pw_rows, ip_rows, strand_rows, ccs_rows, sn_rows], axis=0)
  num_rows = get_total_rows(max_passes)
  rows.set_shape((num_rows, max_length, 1))
  return rows


def process_feature_dict(
    features: Dict[str, Union[np.ndarray, int, bytes]],
    params: Union[config_dict.ConfigDict, config_dict.FrozenConfigDict],
    cap_pw: bool = True,
    cap_ip: bool = True,
    cap_sn: bool = True,
) -> Dict[str, Union[np.ndarray, int, bytes]]:
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
    window_position: The position at which this example starts within the ccs
        read.
    name: Name of the ZMW, e.g. "m64011_181218_235052/315/ccs".
  """
  label = tf.convert_to_tensor(np.array([]))
  subreads = features['subreads']
  num_passes = features['subreads/num_passes']
  rows = format_rows(
      subreads=subreads,
      max_passes=params.max_passes,
      max_length=params.max_length,
      cap_pw=cap_pw,
      cap_ip=cap_ip,
      cap_sn=cap_sn)
  # Don't forget to update DC_FEATURES in dc_constants.py if new features are
  # added/removed.
  features = {
      'rows': rows,
      'label': label,
      'num_passes': num_passes,
      'window_pos': features['window_pos'],
      'name': features['name'],
      'ccs_base_quality_scores': features['ccs_base_quality_scores']
  }
  return features


def parse_example(proto_string: Dict[str, tf.Tensor],
                  inference: bool = False,
                  max_length: int = 120) -> Dict[str, tf.Tensor]:
  """Parses serialized Training or Inference TF.Examples."""
  if inference:
    proto_features = PROTO_FEATURES_INFERENCE
  else:
    proto_features = PROTO_FEATURES_TRAIN
  if not proto_features['ccs_base_quality_scores'].shape:
    proto_features['ccs_base_quality_scores'].shape.append(max_length)
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
) -> Dict[str, tf.Tensor]:
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
    window_position: The position at which this example starts within the ccs
        read.
    name: Name of the ZMW, e.g. "m64011_181218_235052/315/ccs".
  """
  features = parse_example(proto_string, inference, params.max_length)
  flat_subreads = tf.io.decode_raw(features['subreads/encoded'],
                                   dc_constants.TF_DATA_TYPE)
  subreads = tf.reshape(flat_subreads, features['subreads/shape'])
  num_passes = tf.cast(features['subreads/num_passes'],
                       dc_constants.TF_DATA_TYPE)
  if not inference:
    flat_label = tf.io.decode_raw(features['label/encoded'],
                                  dc_constants.TF_DATA_TYPE)
    label = tf.reshape(flat_label, features['label/shape'])

    if params.remove_label_gaps:
      label = remove_internal_gaps_and_shift(label)
    label.set_shape((params.max_length))
  else:
    label = tf.convert_to_tensor(np.array([]))
  rows = format_rows(
      subreads=subreads,
      max_passes=params.max_passes,
      max_length=params.max_length,
      cap_pw=cap_pw,
      cap_ip=cap_ip,
      cap_sn=cap_sn)
  rows = {
      'rows': rows,
      'label': label,
      'num_passes': num_passes,
      'window_pos': features['window_pos'],
      'name': features['name'],
      'ccs_base_quality_scores': features['ccs_base_quality_scores'],
  }
  return rows


def tf_example_to_training_tuple(
    tf_example: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Return only subreads and label."""
  return (tf_example['rows'], tf_example['label'])


def filter_ccs_q_score(tf_example: Dict[str, tf.Tensor],
                       max_phred_qual: int = 0) -> bool:
  """Returns True if phred is less than max_phred_qual."""
  # When max_phred_qual is 0, do not filter.
  if max_phred_qual == 0:
    return True
  phred = utils.tf_avg_phred(tf_example['ccs_base_quality_scores'])
  return phred <= max_phred_qual


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
                example_label_tuple: bool = False) -> tf.data.Dataset:
  """Parses TFRecords and return a dataset.

  Args:
    file_pattern: File path(s) to be parsed by create_glob_list.
    num_epochs: How many epochs for which to repeat.
    batch_size: How many examples should be in each batch.
    params: Hyperparameters used to format the subreads into rows.
    inference: Whether to parse tf.Examples for inference or training.
    cap_pw: If True, pulse width values are capped.
    cap_ip: If True, interpulse distance values are capped.
    limit: Max number of examples to get. Set to -1 for no limit.
    drop_remainder: Passed to TFRecordDataset.batch
    example_label_tuple: If True, output simplified format for training/eval
      as (rows, label)

  Returns:
    A dataset for which each batch has the following elements:
    rows: Input matrix that will be fed into neural networks for training.
    label: Label vector that will be used for training.
    num_passes: The number of subreads present in this example.
    window_position: The position at which this example starts within the ccs
        read.
    name: Name of the ZMW, e.g. "m64011_181218_235052/315/ccs".
  """

  def _process_input_helper(proto_string: tf.Tensor) -> Dict[str, tf.Tensor]:
    return process_input(
        proto_string=proto_string,
        params=params,
        cap_pw=cap_pw,
        cap_ip=cap_ip,
        inference=inference)

  def _filter_q_helper(tf_example: Dict[str, tf.Tensor]) -> bool:
    return filter_ccs_q_score(tf_example, params.max_phred_qual)

  file_patterns = create_glob_list(file_pattern)
  ds = tf.data.TFRecordDataset(file_patterns, compression_type='GZIP')
  ds = ds.map(map_func=_process_input_helper)

  if params.max_phred_qual:
    ds = ds.filter(_filter_q_helper)

  ds = ds.shuffle(buffer_size=params.buffer_size, reshuffle_each_iteration=True)

  if num_epochs:
    ds = ds.repeat(num_epochs)

  # When training, we can drop num_passes, window_position, and name.
  if example_label_tuple:
    ds = ds.map(
        tf_example_to_training_tuple,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
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

  def _process_input_helper(proto_string: tf.Tensor) -> Dict[str, tf.Tensor]:
    # Set inference to False here because we only use this function with
    # tf.Examples that have labels present.
    return process_input(
        proto_string=proto_string,
        params=params,
        inference=False,
        cap_pw=True,
        cap_ip=True)

  def _filter_q_helper(tf_example: Dict[str, tf.Tensor]) -> bool:
    return filter_ccs_q_score(tf_example, params.max_phred_qual)

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

    # Best practices suggest batching first, but this map errors out when we
    # batch first, so do this before mapping.
    ds = ds.map(
        _process_input_helper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    ds = ds.filter(_filter_q_helper, 'ccs_quality_filter')

    if is_training:
      ds = ds.shuffle(
          buffer_size=params['buffer_size'], reshuffle_each_iteration=True)

    # Best practices suggest batching before mapping.
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        tf_example_to_training_tuple,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = ds.take(limit)
    return ds

  return input_fn
