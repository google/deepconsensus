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
"""Methods for DeepConsensus stitch-predictions step."""

import dataclasses
from typing import Iterable, Optional, Tuple
from absl import logging

from deepconsensus.utils import dc_constants
from deepconsensus.utils import utils


@dataclasses.dataclass
class DCModelOutput:
  molecule_name: str
  window_pos: int
  sequence: Optional[str] = None
  quality_string: Optional[str] = None


def get_full_sequence(deepconsensus_outputs: Iterable[DCModelOutput],
                      example_width: int,
                      fill_n: bool = False):
  """Stitch together windows of predictions into a full sequence."""
  # TODO: Check if sorting is still necessary.
  sorted_deepconsensus_outputs = sorted(
      deepconsensus_outputs, key=lambda dc: dc.window_pos)
  # Build up the full sequence from the sorted windows.
  full_sequence_parts = []
  quality_string_parts = []
  start = 0
  for deepconsensus_output in sorted_deepconsensus_outputs:
    # This while loop is used to handle missing windows
    while deepconsensus_output.window_pos > start:
      if not fill_n:
        return None, ''
      else:
        # Add N-base filler for sequences that were unable to be inferred.
        full_sequence_parts.append('N' * example_width)
        empty_quality_scores = [dc_constants.EMPTY_QUAL] * example_width
        empty_quality_string = utils.quality_scores_to_string(
            empty_quality_scores)
        quality_string_parts.append(empty_quality_string)
        start += example_width
    full_sequence_parts.append(deepconsensus_output.sequence)
    quality_string_parts.append(deepconsensus_output.quality_string)
    start += example_width
  full_sequence = ''.join(full_sequence_parts)
  full_quality_string = ''.join(quality_string_parts)
  return full_sequence, full_quality_string


def remove_gaps_and_padding(sequence: str,
                            quality_string: str) -> Tuple[str, str]:
  """Removes gaps/padding and corresponding quality score from outputs."""
  # Remove padding and gaps from the final sequence.
  final_sequence = ''
  final_quality_string = ''
  bases_to_remove = set([dc_constants.GAP_OR_PAD])
  # Only keep bases and quality scores for non padding and non gap positions.
  for base, quality in zip(sequence, quality_string):
    if base not in bases_to_remove:
      final_sequence += base
      final_quality_string += quality

  assert len(final_sequence) == len(final_quality_string)
  assert dc_constants.GAP_OR_PAD not in final_sequence
  return final_sequence, final_quality_string


def is_quality_above_threshold(quality_string, min_quality):
  quality_score_array = utils.quality_string_to_array(quality_string)
  # Round the phred score to ensure expected behavior. Without rounding, a
  # read with all base qualities equal to 10 will have an average phred of
  # 9.99999 due to python floating point precision. Such as read would get
  # filtered out if min_quality is 10.
  rounded_avg_phred = round(utils.avg_phred(quality_score_array), 5)
  logging.vlog(3, 'Quality is %d', rounded_avg_phred)
  return rounded_avg_phred >= min_quality


def format_as_fastq(molecule_name: str, sequence: str,
                    quality_string: str) -> str:
  formatted_for_fastq = f'@{molecule_name}\n'
  formatted_for_fastq += f'{sequence}\n'
  formatted_for_fastq += '+\n'
  formatted_for_fastq += f'{quality_string}\n'
  return formatted_for_fastq


@dataclasses.dataclass
class OutcomeCounter:
  empty_sequence: int = 0
  only_gaps_and_padding: int = 0
  failed_quality_filter: int = 0
  failed_length_filter: int = 0
  success: int = 0


def stitch_to_fastq(molecule_name: str, predictions: Iterable[DCModelOutput],
                    example_width: int, min_quality: int, min_length: int,
                    outcome_counter: OutcomeCounter) -> Optional[str]:
  """Stitch windows of predictions together, filter, and make FASTQ string."""
  full_sequence, full_quality_string = get_full_sequence(
      deepconsensus_outputs=predictions, example_width=example_width)
  # Filter out the read if it is empty after stitching.
  if not full_sequence:
    outcome_counter.empty_sequence += 1
    logging.vlog(1, 'Filtered out read that was empty after stitching: %s',
                 molecule_name)

    return None

  final_sequence, final_quality_string = remove_gaps_and_padding(
      sequence=full_sequence, quality_string=full_quality_string)
  # Filter out the read if it contains only gaps or padding and no bases.
  if not final_sequence:
    outcome_counter.only_gaps_and_padding += 1
    logging.vlog(
        1,
        'Filtered out read that contained only gaps/padding and no bases: %s',
        molecule_name)
    return None

  # Filter out the read if its quality scores are too low.
  if not is_quality_above_threshold(
      quality_string=final_quality_string, min_quality=min_quality):
    outcome_counter.failed_quality_filter += 1
    logging.vlog(1, 'Filtered out read below quality threshold: %s',
                 molecule_name)
    return None

  # Filter out the read if it is too short.
  if len(final_sequence) < min_length:
    outcome_counter.failed_length_filter += 1
    logging.vlog(1, 'Filtered out read below length threshold: %s',
                 molecule_name)
    return None

  fastq = format_as_fastq(
      molecule_name=molecule_name,
      sequence=final_sequence,
      quality_string=final_quality_string)
  outcome_counter.success += 1
  return fastq
