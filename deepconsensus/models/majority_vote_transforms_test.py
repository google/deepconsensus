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
"""Tests for deepconsensus.models.majority_vote_transforms."""

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as beam_testing_util

from deepconsensus.models import majority_vote_transforms
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils


class GetConsensusFromMajorityVoteDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='one read',
          label_bases='ACTG',
          label_expanded_cigar='MMMM',
          subread_bases=['ACTG'],
          subread_expanded_cigars=['MMMM'],
          truth_and_consensus=('ACTG', 'ACTG')),
      dict(
          testcase_name='multiple reads no disagreements',
          label_bases='ACTAG',
          label_expanded_cigar='MMMMM',
          subread_bases=['ACTAG', 'ACTAG', 'ACTAG'],
          subread_expanded_cigars=['MMMMM', 'MMMMM', 'MMMMM'],
          truth_and_consensus=('ACTAG', 'ACTAG')),
      dict(
          testcase_name='multiple reads one disagreements',
          label_bases='ACTAG',
          label_expanded_cigar='MMMMM',
          subread_bases=['ACTA%s' % dc_constants.GAP_OR_PAD, 'ACTAG', 'ACTAG'],
          subread_expanded_cigars=['MMMMD', 'MMMMM', 'MMMMM'],
          truth_and_consensus=('ACTAG', 'ACTAG')),
      dict(
          testcase_name='multiple reads many disagreements',
          label_bases='ACTAGG',
          label_expanded_cigar='MMMMMM',
          subread_bases=[
              'ACT%sGG' % dc_constants.GAP_OR_PAD,
              'ACTAG%s' % dc_constants.GAP_OR_PAD,
              'ACT%sG%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD),
              'ACT%s' % (dc_constants.GAP_OR_PAD * 3)
          ],
          subread_expanded_cigars=[
              'MMM%sMI' % dc_constants.GAP_OR_PAD,
              'MMMIM%s' % dc_constants.GAP_OR_PAD,
              'MMM%sM%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD),
              'MMM%sD%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD)
          ],
          truth_and_consensus=(
              'ACTAGG',
              'ACT%sG%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD))),
      dict(
          testcase_name='ties at each position',
          label_bases='ACTAGA',
          label_expanded_cigar='MMMMMM',
          subread_bases=[
              'AAAAAA',
              'ATCG%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD)
          ],
          subread_expanded_cigars=[
              'MMMMMM',
              'MMMM%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD)
          ],
          truth_and_consensus=('ACTAGA', 'ATCGAA')),
  )
  def test_get_consensus(self, label_bases, label_expanded_cigar, subread_bases,
                         subread_expanded_cigars, truth_and_consensus):
    """Test that consensus sequence correctly determined by majority vote."""

    pws = [[1] * len(label_bases) for _ in subread_bases]
    ips = [[2] * len(label_bases) for _ in subread_bases]
    subread_strand = [deepconsensus_pb2.Subread.REVERSE for _ in subread_bases]
    dc_input = test_utils.make_deepconsensus_input(
        label_bases=label_bases,
        label_expanded_cigar=label_expanded_cigar,
        subread_bases=subread_bases,
        subread_expanded_cigars=subread_expanded_cigars,
        pws=pws,
        ips=ips,
        subread_strand=subread_strand)
    with test_pipeline.TestPipeline() as p:
      filtered_reads = (
          p
          | beam.Create([dc_input])
          | beam.ParDo(
              majority_vote_transforms.GetConsensusFromMajorityVoteDoFn()))

      truth, consensus = truth_and_consensus
      expected = (truth, consensus, dc_input)
      beam_testing_util.assert_that(filtered_reads,
                                    beam_testing_util.equal_to([expected]))


class CountMatchesFromSequenceDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='all matches',
          truth_bases='ATCG',
          consensus='ATCG',
          matches_and_total=(4, 4)),
      dict(
          testcase_name='one mismatch',
          truth_bases='ATCG',
          consensus='ATGG',
          matches_and_total=(3, 4)),
      dict(
          testcase_name='many mismatches',
          truth_bases='ATCG%sGT' % (dc_constants.GAP_OR_PAD * 2),
          consensus='ATGGT%sAT' % dc_constants.GAP_OR_PAD,
          matches_and_total=(4, 6)))
  def test_count_matches_from_sequence(self, truth_bases, consensus,
                                       matches_and_total):
    """Test that matches counted correctly from truth and consensus sequence."""

    # DeepConsensusInput proto not used in this DoFn, but should be returned in
    # case we want to write out examples with errors downstead.
    default_dc_input = test_utils.make_deepconsensus_input()

    with test_pipeline.TestPipeline() as p:
      num_matches_and_positions = (
          p
          | beam.Create([(truth_bases, consensus, default_dc_input)])
          | beam.ParDo(majority_vote_transforms.CountMatchesFromSequenceDoFn()))

      matches, total = matches_and_total
      expected = (matches, total, default_dc_input)
      beam_testing_util.assert_that(num_matches_and_positions,
                                    beam_testing_util.equal_to([expected]))


if __name__ == '__main__':
  absltest.main()
