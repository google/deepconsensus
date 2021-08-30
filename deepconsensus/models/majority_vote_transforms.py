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
"""DoFns for processing PacBio BAMs."""

import collections

from typing import Iterable, Tuple

import apache_beam as beam

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from nucleus.protos import cigar_pb2
from nucleus.util import cigar as cigar_utils

OPS_TO_CONSIDER = frozenset([
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH, cigar_pb2.CigarUnit.SEQUENCE_MATCH,
    cigar_pb2.CigarUnit.INSERT, cigar_pb2.CigarUnit.DELETE,
    cigar_pb2.CigarUnit.SEQUENCE_MISMATCH, cigar_pb2.CigarUnit.CLIP_SOFT
])

OP_CHARS_TO_CONSIDER = frozenset(
    [cigar_utils.CIGAR_OPS_TO_CHAR[op] for op in OPS_TO_CONSIDER])


class GetConsensusFromMajorityVoteDoFn(beam.DoFn):
  """DoFn that yields (label, consensus, and given DeepConsensusInput proto).

  Consensus sequence is computed using the most frequent value at each position
  in the subreads. If there is no subread data at a given position, external
  padding is used as the majority vote.
  """

  def process(
      self, deepconsensus_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[Tuple[str, str, deepconsensus_pb2.DeepConsensusInput]]:
    """Yields label, consensus, and given DeepConsensusInput proto."""

    truth = deepconsensus_input.label
    subreads = deepconsensus_input.subreads
    consensus = ''
    for base_index in range(len(truth.bases)):
      counts = collections.defaultdict(int)

      assert len(subreads) >= 1
      for subread in subreads:
        if len(subread.bases) <= base_index:
          return
        curr_base = str(subread.bases[base_index])
        counts[curr_base] += 1

      # If all reads had padding at this position, counts will be empty. In that
      # case, we can include padding in the consensus sequence because there is
      # no other information available.
      # pylint: disable=cell-var-from-loop
      if not counts:
        curr_majority_vote = dc_constants.GAP_OR_PAD
      else:
        # Counts in E. Coli genome are
        # [('A', 1141538), ('T', 1142980), ('C', 1177641), ('G', 1180362)].
        # Sort based on which bases are most frequent.
        order_by_genomic_freq = f'GCTA{dc_constants.GAP_OR_PAD}'
        sorted_by_genomic_freq = sorted(counts, key=order_by_genomic_freq.index)
        curr_majority_vote = max(
            sorted_by_genomic_freq, key=lambda base: counts[base])

      consensus += curr_majority_vote

    # Ensure that we are using str in Python 2, not unicode.
    yield str(truth.bases), str(consensus), deepconsensus_input


class CountMatchesFromSequenceDoFn(beam.DoFn):
  """DoFn that yields (matches, total positions, and deepconsensus_input).

  The number of matches is the count of positions at which the truth and the
  consensus sequence have the same character. Positions where the label has
  external padding are ignored when counting matches and total positions.
  """

  def __init__(self):
    self.matches_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'num_matches')
    self.positions_counter = beam.metrics.Metrics.counter(
        self.__class__, 'num_positions')

  def process(
      self, molecule_info: Tuple[str, str, deepconsensus_pb2.DeepConsensusInput]
  ) -> Iterable[Tuple[int, int, deepconsensus_pb2.DeepConsensusInput]]:
    """Yields (# of matches, total positions, and given DeepConsensusInput)."""

    truth_bases, consensus, deepconsensus_input = molecule_info
    assert len(truth_bases) == len(consensus)
    num_matches = 0
    num_total = 0
    for i in range(len(truth_bases)):
      # Ignore positions in the label where there is no information.
      if truth_bases[i] != dc_constants.GAP_OR_PAD:
        num_matches += truth_bases[i] == consensus[i]
        num_total += 1
    self.matches_counter.inc(value=num_matches)
    self.positions_counter.inc(value=num_total)
    yield num_matches, num_total, deepconsensus_input


class GetHardExamplesDoFn(beam.DoFn):
  """DoFn that yields examples where majority vote made an error."""

  def process(
      self, matches_total_input: Tuple[int, int,
                                       deepconsensus_pb2.DeepConsensusInput]
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Yields protos containing positions where majority vote made an error."""

    num_matches, num_total, deepconsensus_input = matches_total_input
    if num_matches != num_total:
      yield deepconsensus_input
