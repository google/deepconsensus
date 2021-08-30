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
"""DoFns for stitching together windowed predictions to form full sequence."""

import copy
from typing import Iterable, Tuple

import apache_beam as beam
from apache_beam import metrics

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from deepconsensus.utils import utils


class GetFullSequenceDoFn(beam.DoFn):
  """DoFn that yields the molecule name and full sequence for the molecule."""

  def __init__(self, example_width: int, fill_n=False):
    self.incomplete_sequences_counter = metrics.Metrics.counter(
        self.__class__, 'incomplete_sequences_counter')
    self.example_width = example_width
    self.fill_n = fill_n

  def process(
      self, molecule_name_and_deepconsensus_inputs: Tuple[
          str, Iterable[deepconsensus_pb2.DeepConsensusInput]]
  ) -> Iterable[Tuple[str, str, str]]:
    """Yields the molecule name and full sequence for the molecule."""
    molecule_name, deepconsensus_inputs = molecule_name_and_deepconsensus_inputs
    deepconsensus_inputs_copy = copy.deepcopy(deepconsensus_inputs)
    sorted_deepconsensus_inputs = sorted(
        deepconsensus_inputs_copy, key=lambda dc: dc.molecule_start)
    # Build up the full sequence from the sorted windows.
    full_sequence_parts = []
    quality_string_parts = []
    start = 0
    for deepconsensus_input in sorted_deepconsensus_inputs:
      # This while loop is used to handle missing windows
      while deepconsensus_input.molecule_start > start:
        self.incomplete_sequences_counter.inc()
        if not self.fill_n:
          return
        else:
          # Add N-base filler for sequences that were unable
          # to be inferred.
          full_sequence_parts.append('N' * self.example_width)
          empty_quality_scores = [dc_constants.EMPTY_QUAL] * self.example_width
          empty_quality_string = utils.quality_scores_to_string(
              empty_quality_scores)
          quality_string_parts.append(empty_quality_string)
          start += self.example_width
      full_sequence_parts.append(deepconsensus_input.deepconsensus_prediction)
      quality_string_parts.append(deepconsensus_input.quality_string)
      start += self.example_width
    full_sequence = ''.join(full_sequence_parts)
    full_quality_string = ''.join(quality_string_parts)
    if full_sequence:
      yield (molecule_name, full_sequence, full_quality_string)


class RemoveGapsAndPaddingDoFn(beam.DoFn):
  """DoFn that removes gaps/padding and corresponging quality string char."""

  def process(
      self, name_sequence_scores: Tuple[str, str,
                                        str]) -> Iterable[Tuple[str, str, str]]:
    """Removes gaps/padding and corresponding quality score from outputs."""

    molecule_name, full_sequence, quality_string = name_sequence_scores
    # Remove padding and gaps from the final sequence.
    final_sequence = ''
    final_quality_string = ''
    bases_to_remove = set([dc_constants.GAP_OR_PAD])
    # Only keep bases and quality scores for non padding and non gap positions.
    for base, quality in zip(full_sequence, quality_string):
      if base not in bases_to_remove:
        final_sequence += base
        final_quality_string += quality

    assert len(final_sequence) == len(final_quality_string)
    assert dc_constants.GAP_OR_PAD not in final_sequence
    if final_sequence:
      yield (molecule_name, final_sequence, final_quality_string)


class FilterByQualityDoFn(beam.DoFn):
  """DoFn that yields inputs that meets the specified quality threshold."""

  def __init__(self, min_quality):
    self.min_quality = min_quality
    self.reads_below_min_qual = metrics.Metrics.counter(
        self.__class__, 'reads_below_min_qual_counter')

  def process(
      self, name_sequence_scores: Tuple[str, str,
                                        str]) -> Iterable[Tuple[str, str, str]]:
    """Yields a string for a FASTQ entry, which is contig name and sequence."""
    molecule_name, final_sequence, final_quality_string = name_sequence_scores
    assert dc_constants.GAP_OR_PAD not in final_sequence
    quality_score_array = utils.quality_string_to_array(final_quality_string)
    # Round the phred score to ensure expected behavior. Without rounding, a
    # read with all base qualities equal to 10 will have an average phred of
    # 9.99999 due to python floating point precision. Such as read would get
    # filtered out if min_quality is 10.
    rounded_avg_phred = round(utils.avg_phred(quality_score_array), 5)
    if rounded_avg_phred < self.min_quality:
      self.reads_below_min_qual.inc()
      return
    else:
      yield (molecule_name, final_sequence, final_quality_string)


class FilterByReadLengthDoFn(beam.DoFn):
  """DoFn that yields inputs that meets the specified quality threshold."""

  def __init__(self, min_length):
    self.min_length = min_length

  def process(
      self, name_sequence_scores: Tuple[str, str,
                                        str]) -> Iterable[Tuple[str, str, str]]:
    """Yields a string for a FASTQ entry, which is contig name and sequence."""
    molecule_name, full_sequence, quality_string = name_sequence_scores
    if len(full_sequence) >= self.min_length:
      yield (molecule_name, full_sequence, quality_string)


class ConvertToFastqStrDoFn(beam.DoFn):
  """DoFn that yields a string corresponding to a FASTQ entry."""

  def __init__(self):
    self.total_molecules_written = metrics.Metrics.counter(
        self.__class__, 'total_molecules_written_counter')

  def process(self, name_sequence_scores: Tuple[str, str,
                                                str]) -> Iterable[str]:
    """Yields a string for a FASTQ entry, which is contig name and sequence."""
    molecule_name, final_sequence, final_quality_string = name_sequence_scores
    fragment_name = molecule_name + '/ccs'
    formatted_for_fastq = f'@{fragment_name}\n'
    formatted_for_fastq += f'{final_sequence}\n'
    formatted_for_fastq += '+\n'
    formatted_for_fastq += f'{final_quality_string}\n'
    self.total_molecules_written.inc()
    yield formatted_for_fastq
