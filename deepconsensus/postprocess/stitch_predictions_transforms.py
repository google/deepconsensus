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


class ConvertToFastqStrDoFn(beam.DoFn):
  """DoFn that yields a string corresponding to a FASTQ entry."""

  def process(self, name_sequence_scores: Tuple[str, str,
                                                str]) -> Iterable[str]:
    """Yields a string for a FASTQ entry, which is contig name and sequence."""

    molecule_name, full_sequence, quality_string = name_sequence_scores
    # Remove padding and gaps from the final sequence.
    final_sequence = ''
    final_quality_string = ''
    bases_to_remove = set([dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD])
    # Only keep bases and quality scores for non padding and non gap positions.
    for base, quality in zip(full_sequence, quality_string):
      if base not in bases_to_remove:
        final_sequence += base
        final_quality_string += quality

    assert len(final_sequence) == len(final_quality_string)
    assert dc_constants.GAP_OR_PAD not in final_sequence
    assert dc_constants.GAP_OR_PAD not in final_sequence
    if not full_sequence:
      return
    # Format the molecule name correctly for the FASTQ file.
    fragment_name = molecule_name + '/ccs'
    formatted_for_fastq = f'@{fragment_name}\n'
    formatted_for_fastq += f'{final_sequence}\n'
    formatted_for_fastq += '+\n'
    formatted_for_fastq += f'{final_quality_string}\n'
    yield formatted_for_fastq
