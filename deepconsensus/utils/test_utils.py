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
"""Utilities to help with testing code."""

import os
from typing import Union, Text, List, Tuple

import apache_beam as beam
import numpy as np

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from nucleus.protos import bed_pb2
from nucleus.testing import test_utils
from nucleus.util import struct_utils

DEEPCONSENSUS_DATADIR = ''


def deepconsensus_testdata(filename):
  """Gets the path to filename in testdata.

  These paths are only known at runtime, after flag parsing
  has occurred.

  Args:
    filename: The name of a testdata file in the core genomics testdata
      directory. For example, if you have a test file in
      "DEEPCONSENSUS_DATADIR/deepconsensus/testdata/foo.txt", filename should be
      "foo.txt" to get a path to it.

  Returns:
    The absolute path to a testdata file.
  """
  return test_utils.genomics_testdata(
      os.path.join('deepconsensus/testdata', filename), DEEPCONSENSUS_DATADIR)


def make_read_with_info(expanded_cigar=None,
                        pw=None,
                        ip=None,
                        sn=None,
                        subread_indices=None,
                        subread_strand=None,
                        unsup_insertions_by_pos_keys=None,
                        unsup_insertions_by_pos_values=None,
                        **kwargs):
  """Create read with info fields filled in for testing."""

  read = test_utils.make_read(**kwargs)
  if expanded_cigar is not None:
    struct_utils.set_string_field(read.info, 'expanded_cigar', expanded_cigar)
  if subread_strand == deepconsensus_pb2.Subread.REVERSE:
    read.alignment.position.reverse_strand = True
  elif subread_strand == deepconsensus_pb2.Subread.FORWARD:
    read.alignment.position.reverse_strand = False
  if pw is not None:
    struct_utils.set_int_field(read.info, 'pw', pw)
  if ip is not None:
    struct_utils.set_int_field(read.info, 'ip', ip)
  if sn is not None:
    struct_utils.set_number_field(read.info, 'sn', sn)
  if subread_indices is not None:
    struct_utils.set_int_field(read.info, 'subread_indices', subread_indices)
  if unsup_insertions_by_pos_keys is not None:
    struct_utils.set_int_field(read.info, 'unsup_insertions_by_pos_keys',
                               unsup_insertions_by_pos_keys)
  if unsup_insertions_by_pos_values is not None:
    struct_utils.set_int_field(read.info, 'unsup_insertions_by_pos_values',
                               unsup_insertions_by_pos_values)
  return read


def make_deepconsensus_input(inference: bool = False, **kwargs):
  """Create DeepConsensusInput proto for testing.

  Args:
    inference: whether to generate a proto for inference or training mode.
    **kwargs: Any of the keys in `default_kwargs` can be passed in as named
      arguments for this function. Any values passed in will override the
      defaults present in `default_kwargs`.

  Returns:
    deepconsensus_pb2.DeepConsensusInput proto.
  """
  default_kwargs = {
      'molecule_name': 'm54238_180901_011437/8389007/100_110',
      'molecule_start': 200,
      'subread_strand': [deepconsensus_pb2.Subread.REVERSE],
      'sn': [0.1, 0.2, 0.3, 0.4],
      'subread_bases': ['ATCGA'],
      'subread_expanded_cigars': ['MMMMM'],
      'pws': [[1] * 5],  # Same as len(subreads_bases).
      'ips': [[2] * 5],  # Same as len(pws).
  }
  if not inference:
    default_kwargs.update({
        'chrom_name': 'chr',
        'chrom_start': 1,
        'chrom_end': 6,
        'label_bases': 'ATCGA',
        'label_expanded_cigar': 'MMMMM',
        'label_base_positions': [],
        'strand': bed_pb2.BedRecord.Strand.FORWARD_STRAND,
    })
  default_kwargs.update(**kwargs)

  subread_strand = default_kwargs.pop('subread_strand')
  subread_bases = default_kwargs.pop('subread_bases')
  subread_expanded_cigars = default_kwargs.pop('subread_expanded_cigars')
  pws = default_kwargs.pop('pws')
  ips = default_kwargs.pop('ips')

  if not (len(subread_bases) == len(subread_expanded_cigars) == len(pws) ==
          len(ips) == len(subread_strand)):
    raise ValueError(
        'There must be the same number of entries in `subread_bases`, '
        '`subread_expanded_cigars`, `pws`, `ips`, and `subread_strand`: {}, {}, {} {} {}.'
        .format(subread_bases, subread_expanded_cigars, pws, ips,
                subread_strand))

  for (sb, sec, pw, ip) in zip(subread_bases, subread_expanded_cigars, pws,
                               ips):
    if not len(sb) == len(sec) == len(pw) == len(ip):
      raise ValueError(
          'There must be the same length in each element of `subread_bases`, '
          '`subread_expanded_cigars`, `pws`, and `ips`: {}, {}, {}, {}.'.format(
              sb, sec, pw, ip))

  # Label strand is always forward.
  if not inference:
    label_bases = default_kwargs.pop('label_bases')
    label_expanded_cigar = default_kwargs.pop('label_expanded_cigar')
    label_base_positions = default_kwargs.pop('label_base_positions')
    label = deepconsensus_pb2.Subread(
        molecule_name=default_kwargs['molecule_name'],
        bases=label_bases,
        expanded_cigar=label_expanded_cigar,
        base_positions=label_base_positions,
        subread_strand=deepconsensus_pb2.Subread.FORWARD)
    default_kwargs['label'] = label

  subreads = []
  for bases, expanded_cigar, pw, ip in zip(subread_bases,
                                           subread_expanded_cigars, pws, ips):
    subread = deepconsensus_pb2.Subread(
        molecule_name=default_kwargs['molecule_name'],
        bases=bases,
        expanded_cigar=expanded_cigar,
        subread_strand=subread_strand.pop(0),
        pw=pw,
        ip=ip)
    subreads.append(subread)
  return deepconsensus_pb2.DeepConsensusInput(
      subreads=subreads, **default_kwargs)


def get_beam_counter_value(pipeline_metrics: beam.metrics.metric.MetricResults,
                           namespace: str, counter_name: str) -> int:
  """Returns the value for the given counter name in the input namespace."""
  metric_filter = beam.metrics.metric.MetricsFilter().with_namespace(
      namespace).with_name(counter_name)
  return pipeline_metrics.query(filter=metric_filter)['counters'][0].committed


def get_one_hot(value: Union[int, np.ndarray]) -> np.ndarray:
  """Returns a one-hot vector for a given value."""
  return np.eye(len(dc_constants.VOCAB), dtype=dc_constants.NP_DATA_TYPE)[value]


def seq_to_array(seq: str) -> List[int]:
  return [dc_constants.VOCAB.index(i) for i in seq]


def multiseq_to_array(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG sequences to DC numeric format."""
  return np.array(list(map(seq_to_array, sequences)))


def seq_to_one_hot(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG to one-hot format."""
  result = []
  for seq in sequences:
    result.append(get_one_hot(multiseq_to_array(seq)))
  result = np.squeeze(result)
  return result.astype(dc_constants.NP_DATA_TYPE)


def convert_seqs(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
  """Creates label and associated y_pred tensor.

  Args:
    sequences: string array inputs for label and prediction

  Returns:
    y_true as array
    y_pred_scores as probability array

  """
  y_true, y_pred_scores = sequences
  y_true = multiseq_to_array(y_true).astype(dc_constants.NP_DATA_TYPE)
  y_pred_scores = seq_to_one_hot(y_pred_scores)
  return y_true, y_pred_scores
