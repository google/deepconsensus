# Copyright (c) 2021, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""DoFns for running inference with Beam and writing out predictions."""

import contextlib
import copy
import io
import itertools
import os
import random
from typing import Any, Dict, Iterable, Tuple

import apache_beam as beam
import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import model_utils
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.utils import colab_utils
from deepconsensus.utils import dc_constants
from deepconsensus.utils import utils


class ParseTfExamplesDoFn(beam.DoFn):
  """DoFn that yields parsed fields of a tf.Example."""

  def __init__(self, params, inference):
    self.params = params
    self.inference = inference

  def process(
      self, proto_string: bytes
  ) -> Iterable[Tuple[np.ndarray, np.ndarray, int,
                      deepconsensus_pb2.DeepConsensusInput]]:
    """Parses a serialized tf.Example and returns one input and label."""
    example_info = data_providers.process_input(
        proto_string=proto_string, params=self.params, inference=self.inference)
    rows, label, num_passes, encoded_deepconsensus_input = example_info
    num_passes = int(num_passes.numpy()[0])
    deepconsensus_input = deepconsensus_pb2.DeepConsensusInput.FromString(
        encoded_deepconsensus_input.numpy())
    yield rows.numpy(), label.numpy(), num_passes, deepconsensus_input


def edit_distance(s1: str, s2: str) -> int:
  """Calculates the Levenstein edit distance.

  Edit distance represents the number of insertions, deletions,
  and substitutions required to change s1 to s2. For example,

  CAT -> BAT  = 1
  CAT -> BATS = 2

  Args:
    s1: String 1
    s2: String 2

  Returns:
    The Levenstein edit distance.

  """
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  # Remove all gaps/padding from strings.
  s1 = s1.replace(dc_constants.GAP_OR_PAD, '')
  s1 = s1.replace(dc_constants.GAP_OR_PAD, '')
  s2 = s2.replace(dc_constants.GAP_OR_PAD, '')
  s2 = s2.replace(dc_constants.GAP_OR_PAD, '')

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2 + 1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                   distances_[-1])))
    distances = distances_
  return distances[-1]


def homopolymer_content(seq: str) -> float:
  """Calculates proportion of seq composed of 3+ repeated bases."""
  seq = seq.replace(dc_constants.GAP_OR_PAD, '').strip(dc_constants.GAP_OR_PAD)
  if not seq:
    return 0.0
  run_length_encoding = [len(list(g)) for _, g in itertools.groupby(seq)]
  hcontent = sum([x for x in run_length_encoding if x >= 3]) / len(seq)
  return round(hcontent, 2)


class RunForwardPassDoFn(beam.DoFn):
  """DoFn that runs a forward pass on data and calculates metrics."""

  def __init__(self, checkpoint_path: str, params: ml_collections.ConfigDict,
               inference: bool):
    self.params = params
    self.checkpoint_path = checkpoint_path
    self.inference = inference
    self.model = None

  def setup(self):
    """Initializes model on each worker."""
    self.model = model_utils.get_model(self.params)
    try:
      self.model.load_weights(self.checkpoint_path)
    except AssertionError:
      # Use this approach for models saved in tf.train.Checkpoint format through
      # the custom training loop code.
      checkpoint = tf.train.Checkpoint(model=self.model)
      checkpoint.restore(self.checkpoint_path)

  def process(
      self, example_info: Tuple[np.ndarray, np.ndarray, int,
                                deepconsensus_pb2.DeepConsensusInput]
  ) -> Iterable[Dict[str, Any]]:
    """Yields the input DeepConsensusInput proto with a prediction added in."""
    rows, label, num_passes, dc_input = example_info
    # Add batch dimension to data, since it is expected by model.
    rows = np.expand_dims(rows, 0)
    # In case setup wasn't run.
    if self.model is None:
      self.setup()
    # Output from model is shape (batch_size, window width, vocab size).
    y_pred_scores = self.model(rows)
    # Get rid of batch dimension (batch size is 1) so we can loop over values.
    y_pred = np.squeeze(np.argmax(y_pred_scores, axis=-1))
    y_pred_bases = ''.join([dc_constants.VOCAB[int(token)] for token in y_pred])
    # Cannot mutate input for beam DoFn so make a copy.
    deepconsensus_input_copy = copy.deepcopy(dc_input)
    deepconsensus_input_copy.deepconsensus_prediction = y_pred_bases
    error_prob = np.squeeze(1 - np.max(y_pred_scores, axis=-1))
    quality_scores = -10 * np.log10(error_prob)
    # Round to the nearest integer and cap at max allowed value.
    quality_scores = np.round(quality_scores, decimals=0)
    quality_scores = np.minimum(quality_scores, dc_constants.MAX_QUAL)
    quality_scores = quality_scores.astype(dtype=np.int32)
    quality_string = utils.quality_scores_to_string(quality_scores)
    deepconsensus_input_copy.quality_string = quality_string

    # Calculate additional information for individual record
    record = {}
    if not self.inference:
      label = np.expand_dims(label, 0)
      for metric in model_utils.get_deepconsensus_metrics():
        record[metric.name] = float(metric(label, y_pred_scores))
      record['label_bases'] = dc_input.label.bases
      base_only = colab_utils.remove_gaps(dc_input.label.bases)
      record['label_length'] = len(base_only)
      record['edit_distance'] = edit_distance(dc_input.label.bases,
                                              y_pred_bases)
      record['homopolymer_content'] = homopolymer_content(dc_input.label.bases)
      # Create a variable for sampling examples, allowing 20% to be correct.
      if random.random() > 0.80:
        record['sample'] = True
      else:
        record['sample'] = record['per_example_accuracy'] < 1.0
      record['unsup_insertion_count'] = dc_input.unsup_insertion_count

    # Attach individual record information.
    # <internal>
    record['num_passes'] = num_passes
    record['pred_bases'] = y_pred_bases
    record['dc_proto'] = deepconsensus_input_copy
    yield record


class Stats(beam.CombineFn):
  """Computes statistics on float values."""

  def __init__(self, group: str, metric: str):
    self.group = group
    self.metric = metric

  def create_accumulator(self):
    # Mean, StdDev, Min, Max, Count
    return (0.0, 0.0, float('inf'), float('-inf'), 0)

  def add_input(self, running_values, val):
    (r_sum, r_sumsq, r_min, r_max, r_count) = running_values
    r_sum = r_sum + val
    r_sumsq = r_sumsq + val * val
    r_min = min([val, r_min])
    r_max = max([val, r_max])
    r_count += 1
    return (r_sum, r_sumsq, r_min, r_max, r_count)

  def merge_accumulators(self, accumulators):
    r_sum, r_sumsq, r_min, r_max, r_count = zip(*accumulators)
    return sum(r_sum), sum(r_sumsq), min(r_min), max(r_max), sum(r_count)

  def extract_output(self, running_values):
    (r_sum, r_sumsq, r_min, r_max, r_count) = running_values
    if r_count:
      r_mean = r_sum / r_count
      r_variance = (r_sumsq / r_count) - (r_mean * r_mean)
      r_stddev = np.sqrt(r_variance) if r_variance > 0 else 0
      return {
          'metric': self.metric,
          'group': self.group,
          'mean': r_mean,
          'variance': r_variance,
          'stddev': r_stddev,
          'min': r_min,
          'max': r_max,
          'count': r_count
      }
    else:
      return {
          'metric': self.metric,
          'group': self.group,
          'mean': float('NaN'),
          'variance': float('NaN'),
          'stddev': float('NaN'),
          'count': 0,
          'min': float('NaN'),
          'max': float('NaN')
      }


def update_dict(d: Dict[str, Any], k: Any, v: Any) -> Dict[str, Any]:
  # Used to update values and return result.
  d_out = copy.copy(d)
  d_out[k] = v
  return d_out


class ToCsv(beam.CombineFn):
  """Generates a CSV file from input dictionaries."""

  def create_accumulator(self):
    return []

  def add_input(self, existing, val):
    existing.append(val)
    return existing

  def merge_accumulators(self, accumulators):
    return list(itertools.chain(*accumulators))

  def extract_output(self, rows):
    return pd.DataFrame(rows).to_csv(None, index=False)


class StatsToCsv(beam.PTransform):
  """Calculates stats and integrates stratified key into csv output."""

  def __init__(self, out_dir: str, group: str, metric: str):
    self.out_dir = out_dir
    self.group = group
    self.metric = metric

  def group_all(self, x, group):
    # Used to generate an 'all' total group.
    if group == 'all':
      return 'all'
    else:
      return x[group]

  def expand(self, pcoll):
    fname = f'{self.metric}__{self.group}'

    result = (
        pcoll
        | beam.Map(lambda x: (self.group_all(x, self.group), x[self.metric]))
        | beam.CombinePerKey(Stats(self.group, self.metric))
        | beam.MapTuple(lambda x, y: update_dict(y, 'group_val', str(x)))
        | beam.CombineGlobally(ToCsv())
        | beam.io.textio.WriteToText(
            file_path_prefix=os.path.join(self.out_dir, 'metrics', self.metric,
                                          fname),
            file_name_suffix='.stat.csv',
            shard_name_template=''))
    return result


