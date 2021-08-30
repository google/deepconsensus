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
"""Utility functions used for writing out tf.Examples."""

import json
from typing import Dict, Iterable, List, Optional, Tuple

from apache_beam import metrics
import numpy as np
import tensorflow as tf

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants

from nucleus.protos import bed_pb2
from nucleus.util import sequence_utils


def encode_dna_as_floats(sequence: Iterable[str],
                         vocab: str = dc_constants.VOCAB,
                         offset: int = 0) -> Optional[Iterable[float]]:
  """Encode the sequence as a list of floats using the provided vocab."""

  ids = []
  for base in sequence:
    if base not in vocab:
      return None
    base_id = float(vocab.index(base) + offset)
    ids.append(base_id)
  return ids


def reverse_complement(sequence: str) -> str:
  """Reverses the sequence and complement the valid bases.

  If the complement of a particular base cannot be found in complement_dict, we
  just return the original base. Note: this behavior is specific to
  DeepConsensus and this function should not be used more generally. For a more
  general reverse complementing function, use reverse_complement in
  nucleus/util/sequence_utils.py

  Args:
    sequence: Original sequence of bases.

  Returns:
    The original sequence reversed, with valid bases complemented.
  """
  complement_dict = sequence_utils.DNA_COMPLEMENT
  # If a base is not present in the vocabulary, we don't reverse complement it
  # in this function, and these sequences will get discarded downstream in the
  # TF Example generation pipeline.
  return ''.join(complement_dict.get(nt, nt) for nt in reversed(sequence))


def get_ref_and_start_and_offset(forward_ref_sequence: str,
                                 strand: bed_pb2.BedRecord.Strand,
                                 chrom_start: int,
                                 chrom_end: int) -> Tuple[str, int, int]:
  """Returns information used to help determine label base positions.

  Args:
    forward_ref_sequence: Forward reference sequence corresponding to the region
      covered by dc_input. This may be reverse complemented to match label.
    strand: Either forward or reverse. Cannot be unspecified.
    chrom_start: Genomic start position of this molecule.
    chrom_end: Genomic end position of this molecule.

  Returns:
    ref_sequence: Reference sequence corresponding to the region covered by
      dc_input. May be the reverse complement of input forward_ref_sequence.
    start: Reference position of the first base in the label from dc_input.
    offset: Either +1 or -1, corresponding to whether base positions increase
      or decrease across the label. +1 corresponds to forward strand, whereas
      -1 means the label is for the reverse strand.
  """
  ref_sequence = forward_ref_sequence
  if strand == bed_pb2.BedRecord.Strand.FORWARD_STRAND:
    start = chrom_start
    offset = 1
  elif strand == bed_pb2.BedRecord.Strand.REVERSE_STRAND:
    start = chrom_end
    offset = -1
    # For the reverse strand, we want the reverse complement.
    ref_sequence = reverse_complement(forward_ref_sequence)
  else:
    raise ValueError('Strand must be set.')
  return ref_sequence, start, offset


def get_sequence_without_gaps_or_padding(sequence: str) -> str:
  """Returns the sequence with GAP_OR_PAD and GAP_OR_PAD tokens removed."""
  return sequence.replace(dc_constants.GAP_OR_PAD,
                          '').replace(dc_constants.GAP_OR_PAD, '')


def get_label_start_end(
    label_base_positions: Iterable[int],
    strand: bed_pb2.BedRecord.Strand) -> Tuple[Optional[int], Optional[int]]:
  """Returns start and end coordinates of label in the reference genome.

  Querying the reference genome for these coordinates will produce the label
  sequence. We need to add 1 to either start or end depending on the orientation
  of the reference.

  Args:
    label_base_positions: Reference position of each base in label.
    strand: Either forward or reverse. Cannot be unspecified.

  Returns:
    The coordinates that need to be queried to produce the label sequence with
    gaps and padding removed.
  """
  # Gap and padding tokens may have a position of -1, since they are not
  # actually present in the reference. Remove all instances of -1, since we do
  # not want to consider it when computing min/max position.
  valid_label_base_positions = set(label_base_positions)
  valid_label_base_positions.discard(-1)

  if not valid_label_base_positions:
    return None, None
  start = min(valid_label_base_positions)
  end = max(valid_label_base_positions)
  if strand == bed_pb2.BedRecord.Strand.FORWARD_STRAND:
    end += 1
  elif strand == bed_pb2.BedRecord.Strand.REVERSE_STRAND:
    start -= 1
  else:
    raise ValueError('Strand must be set.')
  return start, end


def get_encoded_subreads_from_example(example):
  """Gets subreads/encoded field from example as a string."""
  return example.features.feature['subreads/encoded'].bytes_list.value[0]


def get_subreads_shape_from_example(example):
  """Gets the subreads/shape field from example as a list of int64."""
  assert len(example.features.feature['subreads/shape'].int64_list.value) == 3
  return example.features.feature['subreads/shape'].int64_list.value[:]


def get_num_passes_from_example(example):
  """Gets the subreads/num_passes field from example as a list of int64."""
  assert len(
      example.features.feature['subreads/num_passes'].int64_list.value) == 1
  return example.features.feature['subreads/num_passes'].int64_list.value[0]


def get_encoded_label_from_example(example):
  """Gets label/encoded field from example as a string."""
  return example.features.feature['label/encoded'].bytes_list.value[0]


def get_label_shape_from_example(example):
  """Gets the label/shape field from example as a list of int64."""
  assert len(example.features.feature['label/shape'].int64_list.value) == 1
  return example.features.feature['label/shape'].int64_list.value[:]


def get_encoded_deepconsensus_input_from_example(example):
  """Gets deepconsensus_input/encoded field from example as a string."""
  return example.features.feature[
      'deepconsensus_input/encoded'].bytes_list.value[0]


def deepconsensus_input_to_example(
    deepconsensus_input: deepconsensus_pb2.DeepConsensusInput,
    example_height: int,
    inference: bool,
    counters: Optional[Dict[str, metrics.Metrics.counter]] = None,
) -> Optional[tf.train.Example]:
  """Returns tf.Example created from the given DeepConsensusInput proto."""
  if not deepconsensus_input.subreads:
    if counters and counters['examples_no_subreads_counter']:
      counters['examples_no_subreads_counter'].inc()
    return

  # Get the example_width from the first subreads.
  example_width = len(deepconsensus_input.subreads[0].bases)

  # The full example will include 4 rows for the signal to noise ratio (sn)
  # values. The remaining rows will contain three sets of per-base values:
  # the base, pulse width (pw), and interpulse distance (ip). Some models
  # may use only a subset of this information downstream.
  per_base_rows = get_per_base_rows(example_height)
  if per_base_rows < 0 or per_base_rows % 4 != 0:
    raise ValueError('example_height - 5 must be non-negative, and divisible '
                     'by four.')
  max_passes = get_max_passes(example_height)

  if len(deepconsensus_input.subreads) > max_passes:
    # Increment a counter if the number of subreads from the
    # deepconsensus_input is more than the `max_passes` derived from the
    # input `example_height`.
    # But still continue.
    if counters and counters['examples_with_discarded_subreads']:
      counters['examples_with_discarded_subreads'].inc()

  example = tf.train.Example()
  features = example.features
  data = np.zeros(
      shape=(example_height, example_width, 1), dtype=dc_constants.NP_DATA_TYPE)
  data += dc_constants.GAP_OR_PAD_INT

  # Number of subreads is capped at num_subreads. In the cases of fewer
  # subreads, rows are left empty.
  kept_subreads = 0
  # Add extra dimension so that shape is (example_width, 1).
  base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices = get_indices(
      max_passes)
  for i in range(min(len(deepconsensus_input.subreads), max_passes)):
    subread = deepconsensus_input.subreads[i]
    # Each tuple should already be padded to the appropriate length.
    assert len(subread.bases) == example_width

    encoded_bases = encode_dna_as_floats(subread.bases)  # pytype: disable=wrong-arg-types
    assert encoded_bases is not None
    data[base_indices[0] + i] += np.expand_dims(np.array(encoded_bases), -1)
    data[pw_indices[0] + i] += np.expand_dims(np.array(subread.pw), -1)
    data[ip_indices[0] + i] += np.expand_dims(np.array(subread.ip), -1)
    data[strand_indices[0] + i] += np.expand_dims(
        np.expand_dims(np.array(subread.subread_strand), -1), -1)
    kept_subreads += 1

  if kept_subreads == 0:
    if counters and counters['examples_no_subreads_counter']:
      counters['examples_no_subreads_counter'].inc()
    return

  if deepconsensus_input.ccs_sequence:
    encoded_ccs_bases = encode_dna_as_floats(deepconsensus_input.ccs_sequence)  # pytype: disable=wrong-arg-types
    data[slice(*ccs_indices)] += np.expand_dims(np.array(encoded_ccs_bases), -1)

  data[slice(*sn_indices)] += np.expand_dims(
      np.expand_dims(np.array(deepconsensus_input.sn), -1), -1)

  features.feature['subreads/encoded'].bytes_list.value.append(data.tostring())
  features.feature['subreads/shape'].int64_list.value.extend(data.shape)
  features.feature['subreads/num_passes'].int64_list.value.append(kept_subreads)

  if not inference:
    label_bases_list = encode_dna_as_floats(deepconsensus_input.label.bases)  # pytype: disable=wrong-arg-types
    assert label_bases_list is not None
    # Final shape of label should be (example_width, ).
    label_matrix = np.array(label_bases_list).astype(dc_constants.NP_DATA_TYPE)
    features.feature['label/encoded'].bytes_list.value.append(
        label_matrix.tostring())
    features.feature['label/shape'].int64_list.value.extend(label_matrix.shape)
  features.feature['deepconsensus_input/encoded'].bytes_list.value.append(
      deepconsensus_input.SerializeToString())
  return example


def between(x, start, end):
  return start <= x and x <= end


def check_region(deepconsensus_input: deepconsensus_pb2.DeepConsensusInput,
                 species: str,
                 contig_chrom: Dict[str, str]) -> Tuple[bool, bool, bool]:
  """Returns whether input should be in eval or train for current species."""

  # Eval set contains only molecules that start and end within the bounds.
  # Train set contains only molecules that are entirely outside of the bounds.
  # Based on this logic, molecules that span the training and eval regions
  # will be thrown out entirely.

  if species == 'ecoli':
    assert 'ecoli' in deepconsensus_input.chrom_name
    in_train_region = between(deepconsensus_input.chrom_start, *
                              dc_constants.ECOLI_REGIONS['TRAIN']) and between(
                                  deepconsensus_input.chrom_end, *
                                  dc_constants.ECOLI_REGIONS['TRAIN'])
    in_eval_region = between(deepconsensus_input.chrom_start, *
                             dc_constants.ECOLI_REGIONS['EVAL']) and between(
                                 deepconsensus_input.chrom_end, *
                                 dc_constants.ECOLI_REGIONS['EVAL'])
    in_test_region = between(deepconsensus_input.chrom_start, *
                             dc_constants.ECOLI_REGIONS['TEST']) and between(
                                 deepconsensus_input.chrom_end, *
                                 dc_constants.ECOLI_REGIONS['TEST'])

  elif species == 'human':
    assert 'ecoli' not in deepconsensus_input.chrom_name
    # Resolve the chrom name for each contig
    chrom_name = contig_chrom.get(deepconsensus_input.chrom_name,
                                  deepconsensus_input.chrom_name)
    in_train_region = chrom_name in dc_constants.HUMAN_TRAIN_REGIONS
    in_eval_region = chrom_name in dc_constants.HUMAN_EVAL_REGIONS
    in_test_region = chrom_name in dc_constants.HUMAN_TEST_REGIONS

  else:
    raise ValueError(
        f"Invalid species: {species}. Must be either 'human' or 'ecoli.'")

  return in_train_region, in_eval_region, in_test_region


def train_eval_partition_fn(
    deepconsensus_input: deepconsensus_pb2.DeepConsensusInput,
    num_partitions: int, species: str, contig_chrom: Dict[str, str]) -> int:
  """Returns an integer for the data split for the given deepconsensus_input."""

  assert num_partitions == 4
  in_train_region, in_eval_region, in_test_region = check_region(
      deepconsensus_input, species, contig_chrom)
  if in_train_region:
    return 0
  elif in_eval_region:
    return 1
  elif in_test_region:
    return 2
  else:
    return 3


def get_empty_columns(
    dc_input: deepconsensus_pb2.DeepConsensusInput) -> List[int]:
  """Returns a list of empty column indices."""
  columns_to_remove = []
  for i in range(len(dc_input.subreads[0].bases)):
    all_internal_gaps = True
    for subread in dc_input.subreads:
      if subread.bases[i] != dc_constants.GAP_OR_PAD:
        all_internal_gaps = False
        break
    if all_internal_gaps:
      columns_to_remove.append(i)
  return columns_to_remove


def pad_bases_pw_ip_cigar(read: deepconsensus_pb2.Subread,
                          padded_len: int) -> None:
  """Add external padding to bases, PW, IP, and cigar."""
  pad_amt = padded_len - len(read.bases)
  if pad_amt > 0:
    str_padding = dc_constants.GAP_OR_PAD * pad_amt
    list_padding = [dc_constants.GAP_OR_PAD_INT] * pad_amt
    read.bases = read.bases + str_padding
    read.pw[:] = list(read.pw) + list_padding
    read.ip[:] = list(read.ip) + list_padding
    read.expanded_cigar = read.expanded_cigar + str_padding


def metrics_to_json(pipeline_result, fname):
  """Collects all counter data and outputs to JSON file."""
  metric_results = pipeline_result.metrics().query()
  results = {}
  for counter in metric_results['counters']:
    counter_name = counter.key.step + ':' + counter.key.metric.name
    results[counter_name] = counter.result
  with tf.io.gfile.GFile(fname, 'w') as f:
    f.write(json.dumps(results, indent=4, sort_keys=True))


def get_indices(max_passes: int) -> Iterable[Tuple[int, int]]:
  """Return row indices for bases/PW/IP/SN in tf.Example subreads array."""
  base_indices = (0, max_passes)
  pw_indices = (max_passes, max_passes * 2)
  ip_indices = (max_passes * 2, max_passes * 3)
  strand_indices = (max_passes * 3, max_passes * 4)
  ccs_indices = (max_passes * 4, max_passes * 4 + 1)
  sn_indices = (max_passes * 4 + 1, max_passes * 4 + 5)
  return base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices


def get_per_base_rows(example_height: int) -> int:
  """Returns the number of rows for bases/PW/IP."""
  return example_height - 5


def get_max_passes(example_height: int) -> int:
  """Returns the max passes for bases/PW/IP."""
  return (example_height - 5) // 4


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
