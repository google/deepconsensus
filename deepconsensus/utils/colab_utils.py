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
"""Utilities for error analysis that can be used in colab."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.utils import dc_constants


WRITE_NORMAL = '\x1b[0m'
WRITE_GREEN_BACKGROUND = '\x1b[102m'
WRITE_RED_BACKGROUND = '\x1b[101m'
WRITE_YELLOW_BACKGROUND = '\x1b[103m'

KMER_SIZE = 10


def remove_gaps(seq: str) -> str:
  """Removes gaps and padding from sequences."""
  seq = seq.replace(dc_constants.GAP_OR_PAD, '')
  return seq


def get_deepconsensus_prediction(
    model: tf.keras.Model, rows: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Runs model on given rows and returns distributions and predictions."""
  softmax_output = model(rows, training=False)
  pred = tf.argmax(softmax_output, axis=-1)
  return softmax_output, pred


def check_has_errors(label: str, pred: str) -> bool:
  """True if there are errors in the prediction, else False."""
  return remove_gaps(label) != remove_gaps(pred)


def ints_to_bases(bases_row: tf.Tensor) -> str:
  """Converts ints to bases based on order in the vocab."""
  return ''.join([dc_constants.VOCAB[int(b)] for b in bases_row])


def convert_to_bases(rows: tf.Tensor, label: tf.Tensor,
                     deepconsensus_pred: tf.Tensor,
                     max_passes: int) -> Tuple[List[str], str, str]:
  """Converts numerical tensors to string of bases."""
  rows = tf.squeeze(rows)
  label = tf.squeeze(label)
  deepconsensus_pred = tf.squeeze(deepconsensus_pred)
  base_indices, _, _, _, _, _ = data_providers.get_indices(max_passes)
  subread_rows_range = range(*base_indices)
  subread_rows = [rows[i, :].numpy() for i in subread_rows_range]
  subread_rows = [row for row in subread_rows if np.sum(row) != 0]
  subread_bases = [ints_to_bases(subread_row) for subread_row in subread_rows]

  label_bases = ints_to_bases(label)
  deepconsensus_pred_bases = ints_to_bases(deepconsensus_pred)
  return subread_bases, label_bases, deepconsensus_pred_bases


def pretty_print_proto(dc_input, print_aux=False):
  """Prints fields from the given DeepConsensusInput proto."""
  spaces = 3 if print_aux else 0
  bases_list = list(str(dc_input.label.bases))
  print('Label:')
  print(''.join([' ' * spaces + base for base in bases_list]))
  print('\n')
  print('Subreads:')
  for read in dc_input.subreads:
    bases_list = list(str(read.bases))
    print(''.join([' ' * spaces + base for base in bases_list]))
  if print_aux:
    print('\n')
    print('PW:')
    for read in dc_input.subreads:
      bases_list = list(str(read.bases))
      print(''.join(['%4d' % value for value in read.pw]))
    print('\n')
    print('IP:')
    for read in dc_input.subreads:
      bases_list = list(str(read.bases))
      print(''.join(['%4d' % value for value in read.ip]))
    print('\n')
    print('Strand:')
    for read in dc_input.subreads:
      print('%4d' % read.subread_strand * len(read.bases))


def get_results_df(experiments: List[int],
                   experiment_pattern: str,
                   decimals: int = 5) -> pd.DataFrame:
  """Returns a dataframe with inference results."""
  all_lines = None
  for experiment in experiments:
    # `experiment_pattern` should contain '{}' that can be filled in with the
    # experiment number.
    inference_csvs = tf.io.gfile.glob(experiment_pattern.format(experiment))
    for inference_csv in inference_csvs:
      n_rows = 2
      curr_df = pd.read_csv(tf.io.gfile.GFile(inference_csv), nrows=n_rows)
      curr_df['experiment_and_work_unit'] = [
          '/'.join(inference_csv.split('/')[-3:-1])
      ] * n_rows
      curr_df['dataset_type'] = 'eval'
      if all_lines is None:
        all_lines = curr_df
      else:
        all_lines = pd.concat([all_lines, curr_df], ignore_index=True)
  assert all_lines is not None
  cols = all_lines.columns.tolist()
  reordered_columns = cols[-2:] + cols[1:-2] + [cols[0]]
  all_lines = all_lines[reordered_columns]
  return all_lines.round(decimals)


def get_results_df_compact(df: pd.DataFrame) -> pd.DataFrame:
  """Returns a compact version of the results with fewer columns."""
  cols_to_keep = [
      'dataset_type', 'experiment_and_work_unit', 'accuracy',
      'per_example_accuracy'
  ]
  return df[cols_to_keep]
