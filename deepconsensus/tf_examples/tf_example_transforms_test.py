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
"""Tests for deepconsensus.tf_examples.tf_example_transforms."""

import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as beam_testing_util
import numpy as np

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_transforms
from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils
from deepconsensus.utils.test_utils import deepconsensus_testdata
from nucleus.io import fasta
from nucleus.protos import bed_pb2
from nucleus.util import ranges
from nucleus.util import sequence_utils


NAMESPACE_BASE = 'deepconsensus.tf_examples.tf_example_transforms'


class FilterNonconfidentRegionsDoFn(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Fully contained in confident regions.',
          dc_input=test_utils.make_deepconsensus_input(
              chrom_name='chr1',
              label_base_positions=range(15, 25),
              strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
          kept=True),
      dict(
          testcase_name='Overlaps confident region, but not fully contained.',
          dc_input=test_utils.make_deepconsensus_input(
              chrom_name='chr1',
              label_base_positions=range(12, 17),
              strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
          kept=False),
      dict(
          testcase_name='No overlap with confident regions.',
          dc_input=test_utils.make_deepconsensus_input(
              chrom_name='chr1',
              label_base_positions=range(0, 5),
              strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
          kept=False),
  )
  def test_filter_nonconfident_regions(self, dc_input, kept):
    """Checks that protos not covered by confident regions are filtered out."""

    truth_bed = deepconsensus_testdata('human/human.truth.bed')
    expected = [dc_input] if kept else []
    with test_pipeline.TestPipeline() as p:
      confident_region_windows = (
          p
          | beam.Create([dc_input])
          | beam.ParDo(
              tf_example_transforms.FilterNonconfidentRegionsDoFn(
                  truth_bed=truth_bed)))
      beam_testing_util.assert_that(confident_region_windows,
                                    beam_testing_util.equal_to(expected))


class AddLabelBasesPositionDoFnTest(parameterized.TestCase):

  def _label_positions_correct(self, reference_fasta):

    def _check_dc_input_proto(dc_input_protos):
      for dc_input in dc_input_protos:
        label_sequence = tf_example_utils.get_sequence_without_gaps_or_padding(
            dc_input.label.bases)
        start, end = tf_example_utils.get_label_start_end(
            dc_input.label.base_positions, dc_input.strand)
        if start is None or end is None:
          return

        region = ranges.make_range(dc_input.chrom_name, start, end)
        reference_reader = fasta.IndexedFastaReader(reference_fasta)
        with reference_reader:
          reference_bases = reference_reader.query(region)
        all_seqs = set()
        all_seqs.add(reference_bases)
        all_seqs.add(label_sequence)
        all_seqs.add(sequence_utils.reverse_complement(reference_bases))
        # label_sequence should be the same as forward or reverse reference
        # sequence.
        self.assertLen(all_seqs, 2)
        break

    return _check_dc_input_proto

  @parameterized.parameters(True, False)
  def test_add_label_positions(self, use_smaller_width_do_fn):
    """Test that label positions match what is in the reference genome."""

    with test_pipeline.TestPipeline() as p:
      dc_input_path = deepconsensus_testdata(
          'human/output/deepconsensus/deepconsensus*.tfrecords.gz')
      reference_fasta = deepconsensus_testdata('human/human.ref.fa.gz')

      dc_input_protos = (
          p
          | beam.io.ReadFromTFRecord(
              dc_input_path,
              coder=beam.coders.ProtoCoder(
                  deepconsensus_pb2.DeepConsensusInput),
              compression_type=CompressionTypes.GZIP)
          | beam.ParDo(
              tf_example_transforms.AddLabelBasesPositionDoFn(
                  reference_fasta=reference_fasta)))

      # Run with and withoout SmallerWidthDoFn here to make sure positions are
      # correct after windowing.
      if use_smaller_width_do_fn:
        example_width = 100
        dc_input_protos = (
            dc_input_protos
            | beam.ParDo(
                tf_example_transforms.GetSmallerWindowDoFn(
                    example_width=example_width, inference=False)))
      beam_testing_util.assert_that(
          dc_input_protos, self._label_positions_correct(reference_fasta))


class GetSmallerWindowDoFn(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Windows with and without external padding',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              molecule_start=0,
              subread_bases=['ATCGTTGC'],
              subread_expanded_cigars=['MMMMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6, 7, 8]],
              pws=[[9, 10, 11, 12, 13, 14, 15, 16]],
              label_bases='ATCGTTGT',
              label_expanded_cigar='MMMMMMMM',
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
                  label_bases='ATC',
                  label_expanded_cigar='MMM'),
              test_utils.make_deepconsensus_input(
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]],
                  label_bases='GTT',
                  label_expanded_cigar='MMM'),
              test_utils.make_deepconsensus_input(
                  molecule_start=6,
                  subread_bases=['GC%s' % dc_constants.GAP_OR_PAD],
                  subread_expanded_cigars=['MM%s' % dc_constants.GAP_OR_PAD],
                  ips=[[7, 8, dc_constants.GAP_OR_PAD_INT]],
                  pws=[[15, 16, dc_constants.GAP_OR_PAD_INT]],
                  label_bases='GT%s' % dc_constants.GAP_OR_PAD,
                  label_expanded_cigar='MM%s' % dc_constants.GAP_OR_PAD)
          ]),
      dict(
          testcase_name='Windows with only padding/gaps in subreads or label',
          example_width=1,
          dc_input=test_utils.make_deepconsensus_input(
              molecule_start=0,
              subread_bases=[
                  'A%s%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD,
                               dc_constants.GAP_OR_PAD),
                  'ATT%s' % dc_constants.GAP_OR_PAD,
              ],
              subread_expanded_cigars=[
                  'M%s%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD,
                               dc_constants.GAP_OR_PAD),
                  'MMM%s' % dc_constants.GAP_OR_PAD,
              ],
              ips=[[
                  1, dc_constants.GAP_OR_PAD_INT, dc_constants.GAP_OR_PAD_INT,
                  dc_constants.GAP_OR_PAD_INT
              ], [1, 2, 3, dc_constants.GAP_OR_PAD_INT]],
              pws=[[
                  4, dc_constants.GAP_OR_PAD_INT, dc_constants.GAP_OR_PAD_INT,
                  dc_constants.GAP_OR_PAD_INT
              ], [5, 6, 7, dc_constants.GAP_OR_PAD_INT]],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
              label_bases='%sATG' % dc_constants.GAP_OR_PAD,
              label_expanded_cigar='%sMMM' % dc_constants.GAP_OR_PAD,
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  molecule_start=0,
                  subread_bases=['A', 'A'],
                  subread_expanded_cigars=['M', 'M'],
                  ips=[[1], [1]],
                  pws=[[4], [5]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
                  label_bases=dc_constants.GAP_OR_PAD,
                  label_expanded_cigar=dc_constants.GAP_OR_PAD),
              test_utils.make_deepconsensus_input(
                  molecule_start=1,
                  subread_bases=[dc_constants.GAP_OR_PAD, 'T'],
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD, 'M'],
                  ips=[[dc_constants.GAP_OR_PAD_INT], [2]],
                  pws=[[dc_constants.GAP_OR_PAD_INT], [6]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
                  label_bases='A',
                  label_expanded_cigar='M'),
              test_utils.make_deepconsensus_input(
                  molecule_start=2,
                  subread_bases=[dc_constants.GAP_OR_PAD, 'T'],
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD, 'M'],
                  ips=[[dc_constants.GAP_OR_PAD_INT], [3]],
                  pws=[[dc_constants.GAP_OR_PAD_INT], [7]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
                  label_bases='T',
                  label_expanded_cigar='M'),
              test_utils.make_deepconsensus_input(
                  molecule_start=3,
                  subread_bases=[dc_constants.GAP_OR_PAD] * 2,
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD] * 2,
                  ips=[[dc_constants.GAP_OR_PAD_INT],
                       [dc_constants.GAP_OR_PAD_INT]],
                  pws=[[dc_constants.GAP_OR_PAD_INT],
                       [dc_constants.GAP_OR_PAD_INT]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
                  label_bases='G',
                  label_expanded_cigar='M'),
          ]),
      dict(
          testcase_name='Unsupported insertion at beg of window',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
              label_bases='ATCGGTT',
              label_expanded_cigar='MMMMIMM',
              subread_indices=[1, 2, 3, 5, 6, 7],
              unsup_insertions_by_pos={3: 1},
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
                  label_bases='ATC',
                  label_expanded_cigar='MMM'),
              test_utils.make_deepconsensus_input(
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]],
                  label_bases='GGTT',
                  label_expanded_cigar='MIMM',
                  unsup_insertion_count=1),
          ]),
      dict(
          testcase_name='Unsupported insertion at beg of molecule',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
              label_bases='AATCGTT',
              label_expanded_cigar='MIMMMMM',
              subread_indices=[2, 3, 4, 5, 6, 7],
              unsup_insertions_by_pos={0: 1},
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
                  label_bases='AATC',
                  label_expanded_cigar='MIMM',
                  unsup_insertion_count=1),
              test_utils.make_deepconsensus_input(
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]],
                  label_bases='GTT',
                  label_expanded_cigar='MMM'),
          ]),
      dict(
          testcase_name='Multiple unsupported insertions at beg of molecule',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
              label_bases='AAATCGTT',
              label_expanded_cigar='MIIMMMMM',
              subread_indices=[3, 4, 5, 6, 7, 8],
              unsup_insertions_by_pos={0: 2},
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
                  label_bases='AAATC',
                  label_expanded_cigar='MIIMM',
                  unsup_insertion_count=2),
              test_utils.make_deepconsensus_input(
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]],
                  label_bases='GTT',
                  label_expanded_cigar='MMM'),
          ]),
  )
  def test_smaller_windows_train(self, example_width, dc_input, expected):
    """Test that examples correctly modified based on specified width."""
    inference = False
    with test_pipeline.TestPipeline() as p:
      windowed = (
          p
          | beam.Create([dc_input])
          | beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width, inference=inference)))
      beam_testing_util.assert_that(windowed,
                                    beam_testing_util.equal_to(expected))

    # Also test when input is a tf.Example.
    # For this test to work, example_height need to be >= #subreads and
    # example_height - 5 should be divisible by 3.
    max_passes = 30
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    with test_pipeline.TestPipeline() as p:
      windowed = (
          p
          | beam.Create([
              tf_example_utils.deepconsensus_input_to_example(
                  dc_input, example_height, inference=inference)
          ])
          | beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width, proto_class='Example', inference=inference)))
      beam_testing_util.assert_that(windowed,
                                    beam_testing_util.equal_to(expected))

  @parameterized.named_parameters(
      dict(
          testcase_name='Windows with and without external padding',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              inference=True,
              molecule_start=0,
              subread_bases=['ATCGTTGC'],
              subread_expanded_cigars=['MMMMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6, 7, 8]],
              pws=[[9, 10, 11, 12, 13, 14, 15, 16]],
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]]),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]]),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=6,
                  subread_bases=['GC%s' % dc_constants.GAP_OR_PAD],
                  subread_expanded_cigars=['MM%s' % dc_constants.GAP_OR_PAD],
                  ips=[[7, 8, dc_constants.GAP_OR_PAD_INT]],
                  pws=[[15, 16, dc_constants.GAP_OR_PAD_INT]]),
          ]),
      dict(
          testcase_name='Windows with only padding/gaps in subreads or label',
          example_width=1,
          dc_input=test_utils.make_deepconsensus_input(
              inference=True,
              molecule_start=0,
              subread_bases=[
                  'A%s%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD,
                               dc_constants.GAP_OR_PAD),
                  'ATT%s' % dc_constants.GAP_OR_PAD,
              ],
              subread_expanded_cigars=[
                  'M%s%s%s' % (dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD,
                               dc_constants.GAP_OR_PAD),
                  'MMM%s' % dc_constants.GAP_OR_PAD,
              ],
              ips=[[
                  1, dc_constants.GAP_OR_PAD_INT, dc_constants.GAP_OR_PAD_INT,
                  dc_constants.GAP_OR_PAD_INT
              ], [1, 2, 3, dc_constants.GAP_OR_PAD_INT]],
              pws=[[
                  4, dc_constants.GAP_OR_PAD_INT, dc_constants.GAP_OR_PAD_INT,
                  dc_constants.GAP_OR_PAD_INT
              ], [5, 6, 7, dc_constants.GAP_OR_PAD_INT]],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2,
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=0,
                  subread_bases=['A', 'A'],
                  subread_expanded_cigars=['M', 'M'],
                  ips=[[1], [1]],
                  pws=[[4], [5]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=1,
                  subread_bases=[dc_constants.GAP_OR_PAD, 'T'],
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD, 'M'],
                  ips=[[dc_constants.GAP_OR_PAD_INT], [2]],
                  pws=[[dc_constants.GAP_OR_PAD_INT], [6]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=2,
                  subread_bases=[dc_constants.GAP_OR_PAD, 'T'],
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD, 'M'],
                  ips=[[dc_constants.GAP_OR_PAD_INT], [3]],
                  pws=[[dc_constants.GAP_OR_PAD_INT], [7]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=3,
                  subread_bases=[dc_constants.GAP_OR_PAD] * 2,
                  subread_expanded_cigars=[dc_constants.GAP_OR_PAD] * 2,
                  ips=[[dc_constants.GAP_OR_PAD_INT],
                       [dc_constants.GAP_OR_PAD_INT]],
                  pws=[[dc_constants.GAP_OR_PAD_INT],
                       [dc_constants.GAP_OR_PAD_INT]],
                  subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
          ]),
      dict(
          testcase_name='Unsupported insertion at beg of window',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              inference=True,
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
              ),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]],
              ),
          ]),
      dict(
          testcase_name='Unsupported insertion at beg of molecule',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              inference=True,
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]]),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]]),
          ]),
      dict(
          testcase_name='Multiple unsupported insertions at beg of molecule',
          example_width=3,
          dc_input=test_utils.make_deepconsensus_input(
              inference=True,
              molecule_start=0,
              subread_bases=['ATCGTT'],
              subread_expanded_cigars=['MMMMMM'],
              ips=[[1, 2, 3, 4, 5, 6]],
              pws=[[9, 10, 11, 12, 13, 14]],
          ),
          expected=[
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=0,
                  subread_bases=['ATC'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[1, 2, 3]],
                  pws=[[9, 10, 11]],
              ),
              test_utils.make_deepconsensus_input(
                  inference=True,
                  molecule_start=3,
                  subread_bases=['GTT'],
                  subread_expanded_cigars=['MMM'],
                  ips=[[4, 5, 6]],
                  pws=[[12, 13, 14]]),
          ]),
  )
  def test_smaller_windows_inference(self, example_width, dc_input, expected):
    """Test that examples correctly modified based on specified width."""
    inference = True
    with test_pipeline.TestPipeline() as p:
      windowed = (
          p
          | beam.Create([dc_input])
          | beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width, inference=inference)))
      beam_testing_util.assert_that(windowed,
                                    beam_testing_util.equal_to(expected))

    # Also test when input is a tf.Example.
    # For this test to work, example_height need to be >= #subreads and
    # example_height - 5 should be divisible by 3.
    max_passes = 30
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    with test_pipeline.TestPipeline() as p:
      windowed = (
          p
          | beam.Create([
              tf_example_utils.deepconsensus_input_to_example(
                  dc_input, example_height, inference=inference)
          ])
          | beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width, proto_class='Example', inference=inference)))
      beam_testing_util.assert_that(windowed,
                                    beam_testing_util.equal_to(expected))


class FilterVariantWindowsDoFnTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='no slack', expected_start_index=2, slack=None),
      dict(testcase_name='some slack', expected_start_index=4, slack=5),
  )
  def test_filter_variants_with_and_without_slack(self, expected_start_index,
                                                  slack):
    """Tests that examples containing variant positions filtered out."""

    # This file contains the following variants, noted as chr_start:ref->alt.
    # variant 1: chr1_1:A->ATATT (end position is 1)
    # variant 2: chr1_18:A->t (end position is 18)
    truth_vcf = test_utils.deepconsensus_testdata('human/human.variants.vcf.gz')
    dc_input = [
        # Includes variant 2, with and without slack.
        test_utils.make_deepconsensus_input(
            chrom_name='chr1',
            label_base_positions=list(range(15, 25)),
            strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
        # Includes variant 1, with and without slack.
        test_utils.make_deepconsensus_input(
            chrom_name='chr1',
            label_base_positions=list(range(9, -1, -1)),
            strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND),
        # Includes variant 1, only with slack.
        test_utils.make_deepconsensus_input(
            chrom_name='chr1',
            label_base_positions=list(range(9, 1, -1)),
            strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND),
        # Includes variant 2, only with slack.
        test_utils.make_deepconsensus_input(
            chrom_name='chr1',
            label_base_positions=list(range(10, 15)),
            strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
        # Includes neither, with and without slack.
        test_utils.make_deepconsensus_input(
            chrom_name='chr1',
            label_base_positions=list(range(30, 26, -1)),
            strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND),
    ]
    expected = dc_input[expected_start_index:]
    with test_pipeline.TestPipeline() as p:
      windows_without_variants = (
          p
          | beam.Create(dc_input)
          | beam.ParDo(
              tf_example_transforms.FilterVariantWindowsDoFn(
                  truth_vcf=truth_vcf, slack=slack)))
      beam_testing_util.assert_that(windows_without_variants,
                                    beam_testing_util.equal_to(expected))


class CreateTrainEvalDoFnBySpeciesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='ecoli train set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=0, chrom_end=5, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=464250, chrom_end=464255, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=500000, chrom_end=500005, chrom_name='ecoli'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=500000, chrom_end=500005, chrom_name='ecoli'),
          ],
          filter_set='train',
          species='ecoli',
      ),
      dict(
          testcase_name='ecoli eval set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=0, chrom_end=5, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=464250, chrom_end=464255, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=500000, chrom_end=500005, chrom_name='ecoli'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=0, chrom_end=5, chrom_name='ecoli'),
          ],
          filter_set='eval',
          species='ecoli',
      ),
      dict(
          testcase_name='ecoli test set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=0, chrom_end=5, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=464250, chrom_end=464255, chrom_name='ecoli'),
              test_utils.make_deepconsensus_input(
                  chrom_start=4178271, chrom_end=4178272, chrom_name='ecoli'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=4178271, chrom_end=4178272, chrom_name='ecoli'),
          ],
          filter_set='test',
          species='ecoli',
      ),
      dict(
          testcase_name='human train set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=8, chrom_end=10, chrom_name='chr1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chrX'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='X'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=8, chrom_end=10, chrom_name='chr1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chrX'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='X'),
          ],
          filter_set='train',
          species='human',
      ),
      dict(
          testcase_name='human eval set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=8, chrom_end=10, chrom_name='chr1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chrX'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='X'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='21'),
          ],
          filter_set='eval',
          species='human',
      ),
      dict(
          testcase_name='human test set',
          deepconsensus_input=[
              test_utils.make_deepconsensus_input(
                  chrom_start=8, chrom_end=10, chrom_name='chr1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='1'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='21'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chrX'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='X'),
          ],
          expected=[
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='chr20'),
              test_utils.make_deepconsensus_input(
                  chrom_start=3, chrom_end=5, chrom_name='20'),
          ],
          filter_set='test',
          species='human',
      ),
  )
  def test_create_train_eval_sets(self, deepconsensus_input, expected,
                                  filter_set, species):
    """Tests that examples correctly assigned to train/eval sets."""

    with test_pipeline.TestPipeline() as p:
      filtered_reads = (
          p
          | beam.Create(deepconsensus_input)
          | beam.ParDo(
              tf_example_transforms.CreateExamplesDoFn(
                  filter_set, species=species)))

      beam_testing_util.assert_that(filtered_reads,
                                    beam_testing_util.equal_to(expected))


class PadExamplesDoFnTest(parameterized.TestCase):

  def _check_padded_len_wrapper(self, padded_len, inference):

    def _check_padded_len(dc_inputs):
      for dc_input in dc_inputs:
        if not inference:
          self.assertLen(dc_input.label.bases, padded_len)
        for subread in dc_input.subreads:
          self.assertLen(subread.bases, padded_len)
          self.assertLen(subread.pw, padded_len)
          self.assertLen(subread.ip, padded_len)
          self.assertLen(subread.expanded_cigar, padded_len)

    return _check_padded_len

  @parameterized.named_parameters(
      dict(
          testcase_name='human eval set',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              subread_bases=['A TCGA -'] * 3,
              subread_expanded_cigars=['M MMMM -'] * 3,
              label_bases='ATTCGAA-',
              label_expanded_cigar='MMMMMMM-',
              pws=[[5, 1, 5, 5, 5, 5, 1, 0]] * 3,
              ips=[[6, 1, 6, 6, 6, 6, 1, 0]] * 3,
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 3,
          ),
          padded_len=10,
      ),)
  def test_pad_examples_small_inputs_train(self, deepconsensus_input,
                                           padded_len):
    """Tests that examples are correctly padded for toy inputs."""
    inference = False
    with test_pipeline.TestPipeline() as p:
      dc_inputs = (
          p
          | 'create_dc_input' >> beam.Create([deepconsensus_input])
          | 'pad' >> beam.ParDo(
              tf_example_transforms.PadExamplesDoFn(
                  padded_len=padded_len, inference=inference)))
      beam_testing_util.assert_that(
          dc_inputs,
          self._check_padded_len_wrapper(padded_len, inference=inference))

  @parameterized.named_parameters(
      dict(
          testcase_name='small example for inference',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              subread_bases=['A TCGA -'] * 3,
              subread_expanded_cigars=['M MMMM -'] * 3,
              pws=[[5, 1, 5, 5, 5, 5, 1, 0]] * 3,
              ips=[[6, 1, 6, 6, 6, 6, 1, 0]] * 3,
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 3,
          ),
          padded_len=10,
      ),)
  def test_pad_examples_small_inputs_inferece(self, deepconsensus_input,
                                              padded_len):
    """Tests that examples are correctly padded for toy inputs."""
    inference = True
    with test_pipeline.TestPipeline() as p:
      dc_inputs = (
          p
          | 'create_dc_input' >> beam.Create([deepconsensus_input])
          | 'pad' >> beam.ParDo(
              tf_example_transforms.PadExamplesDoFn(
                  padded_len=padded_len, inference=inference)))
      beam_testing_util.assert_that(
          dc_inputs,
          self._check_padded_len_wrapper(padded_len, inference=inference))

  @parameterized.parameters([False, True])
  def test_pad_examples_large_inputs(self, inference):
    """Tests that examples are correctly padded for testdata."""
    with test_pipeline.TestPipeline() as p:
      output_path = 'inference_output' if inference else 'output'
      deepconsensus_input_path = deepconsensus_testdata(f'human/{output_path}')
      padded_len = 110

      dc_inputs = (
          p
          | 'read_dc_input' >> beam.io.ReadFromTFRecord(
              os.path.join(deepconsensus_input_path,
                           'deepconsensus/deepconsensus*.tfrecords.gz'),
              coder=beam.coders.ProtoCoder(
                  deepconsensus_pb2.DeepConsensusInput),
              compression_type=CompressionTypes.GZIP)
          | 'chunk_windows' >> beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width=padded_len - 10, inference=inference))
          | 'pad' >> beam.ParDo(
              tf_example_transforms.PadExamplesDoFn(
                  padded_len=padded_len, inference=inference)))
      beam_testing_util.assert_that(
          dc_inputs,
          self._check_padded_len_wrapper(padded_len, inference=inference))


class RemoveSequencesWithInvalidBasesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='only valid bases train',
          bases='ATCG  ',
          should_be_kept=True,
          inference=False,
      ),
      dict(
          testcase_name='only valid bases inference',
          bases='ATCG  ',
          should_be_kept=True,
          inference=True,
      ),
      dict(
          testcase_name='only invalid bases train',
          bases='MNOPQR',
          should_be_kept=False,
          inference=False,
      ),
      dict(
          testcase_name='only invalid bases inference',
          bases='MNOPQR',
          should_be_kept=False,
          inference=True,
      ),
      dict(
          testcase_name='valid and invalid bases train',
          bases='ATCGMN',
          inference=False,
          should_be_kept=False,
      ),
      dict(
          testcase_name='valid and invalid bases inference',
          bases='ATCGMN',
          inference=True,
          should_be_kept=False,
      ),
  )
  def test_remove_empty_subreads(self, bases, should_be_kept, inference):
    """Tests that subreads with invalid bases are removed from the dc_input."""
    with test_pipeline.TestPipeline() as p:
      dc_input = deepconsensus_pb2.DeepConsensusInput(
          subreads=[deepconsensus_pb2.Subread(bases=bases)])
      actual = (
          p
          | beam.Create([dc_input])
          | beam.ParDo(
              tf_example_transforms.RemoveSequencesWithInvalidBasesDoFn(
                  inference=inference)))
      subreads = dc_input.subreads if should_be_kept else []
      expected = [deepconsensus_pb2.DeepConsensusInput(subreads=subreads)]
      beam_testing_util.assert_that(actual,
                                    beam_testing_util.equal_to(expected))


class ConvertToTfExamplesDoFnTest(parameterized.TestCase):

  def _tensorflow_example_is_valid(self, expected_deepconsensus_input,
                                   expected_subreads, expected_subreads_shape,
                                   expected_num_passes, expected_label,
                                   expected_label_shape, inference):

    def _equal(actual):
      subreads_string = tf_example_utils.get_encoded_subreads_from_example(
          actual[0])
      subreads_shape = tf_example_utils.get_subreads_shape_from_example(
          actual[0])
      num_passes = tf_example_utils.get_num_passes_from_example(actual[0])
      num_passes = tf_example_utils.get_num_passes_from_example(actual[0])
      deepconsensus_input = tf_example_utils.get_encoded_deepconsensus_input_from_example(
          actual[0])
      self.assertEqual(subreads_string, expected_subreads.tostring())
      self.assertEqual(subreads_shape, expected_subreads_shape)
      self.assertEqual(num_passes, expected_num_passes)
      self.assertEqual(deepconsensus_input,
                       expected_deepconsensus_input.SerializeToString())
      if not inference:
        label_string = tf_example_utils.get_encoded_label_from_example(
            actual[0])
        label_shape = tf_example_utils.get_label_shape_from_example(actual[0])
        self.assertEqual(label_string, expected_label.tostring())
        self.assertEqual(label_shape, expected_label_shape)

    return _equal

  @parameterized.named_parameters(
      dict(
          testcase_name='equal subreads and height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              subread_bases=['ATCG', 'ATCG'],
              subread_expanded_cigars=['MMMM', 'MMMM'],
              pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
              ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[
                  deepconsensus_pb2.Subread.REVERSE,
                  deepconsensus_pb2.Subread.FORWARD
              ]),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'),
              test_utils.seq_to_array('ATCG'), [1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0], [deepconsensus_pb2.Subread.REVERSE] * 4,
              [deepconsensus_pb2.Subread.FORWARD] * 4, [0.0] * 4, [0.1] * 4,
              [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=2,
          expected_num_passes=2,
          expected_label=np.array(test_utils.seq_to_array('ATCG')),
          expected_label_shape=[4]),
      dict(
          testcase_name='fewer subreads than height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              subread_bases=['ATCG'],
              subread_expanded_cigars=['MMMM'],
              pws=[[1, 2, 3, 4]],
              ips=[[5, 6, 7, 8]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE]),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'),
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [1.0, 2.0, 3.0, 4.0],
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [5.0, 6.0, 7.0, 8.0],
              [float(dc_constants.GAP_OR_PAD_INT)] * 4,
              [deepconsensus_pb2.Subread.REVERSE] * 4,
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [0.0] * 4, [0.1] * 4,
              [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=2,
          expected_num_passes=1,
          expected_label=np.array(test_utils.seq_to_array('ATCG')),
          expected_label_shape=[4]),
      dict(
          testcase_name='more subreads than height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              subread_bases=['ATCG', 'ATCG'],
              subread_expanded_cigars=['MMMM', 'MMMM'],
              pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
              ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'), [1.0, 2.0, 3.0, 4.0],
              [9.0, 10.0, 11.0, 12.0], [deepconsensus_pb2.Subread.REVERSE] * 4,
              [0.0] * 4, [0.1] * 4, [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=1,
          expected_num_passes=1,
          expected_label=np.array(test_utils.seq_to_array('ATCG')),
          expected_label_shape=[4]),
  )
  def test_convert_to_tf_example_train(self, deepconsensus_input, max_passes,
                                       expected_subreads, expected_num_passes,
                                       expected_label, expected_label_shape):
    """Check that tensorflow examples are correctly generated."""
    inference = False
    with test_pipeline.TestPipeline() as p:
      example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
      tf_examples = (
          p
          | 'create_data' >> beam.Create([deepconsensus_input])
          | 'convert_to_tf_examples' >> beam.ParDo(
              tf_example_transforms.ConvertToTfExamplesDoFn(
                  example_height=example_height, inference=inference)))

      # Cast expected subreads and labels to correct data type.
      expected_subreads_shape = list(expected_subreads.shape) + [1]
      beam_testing_util.assert_that(
          tf_examples,
          self._tensorflow_example_is_valid(
              deepconsensus_input,
              expected_subreads.astype(dc_constants.NP_DATA_TYPE),
              expected_subreads_shape,
              expected_num_passes,
              expected_label.astype(dc_constants.NP_DATA_TYPE),
              expected_label_shape,
              inference=inference))

  @parameterized.named_parameters(
      dict(
          testcase_name='equal subreads and height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              subread_bases=['ATCG', 'ATCG'],
              subread_expanded_cigars=['MMMM', 'MMMM'],
              pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
              ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[
                  deepconsensus_pb2.Subread.REVERSE,
                  deepconsensus_pb2.Subread.FORWARD
              ]),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'),
              test_utils.seq_to_array('ATCG'), [1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0], [deepconsensus_pb2.Subread.REVERSE] * 4,
              [deepconsensus_pb2.Subread.FORWARD] * 4, [0.0] * 4, [0.1] * 4,
              [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=2,
          expected_num_passes=2),
      dict(
          testcase_name='fewer subreads than height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              subread_bases=['ATCG'],
              subread_expanded_cigars=['MMMM'],
              pws=[[1, 2, 3, 4]],
              ips=[[5, 6, 7, 8]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE]),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'),
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [1.0, 2.0, 3.0, 4.0],
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [5.0, 6.0, 7.0, 8.0],
              [float(dc_constants.GAP_OR_PAD_INT)] * 4,
              [deepconsensus_pb2.Subread.REVERSE] * 4,
              [float(dc_constants.GAP_OR_PAD_INT)] * 4, [0.0] * 4, [0.1] * 4,
              [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=2,
          expected_num_passes=1),
      dict(
          testcase_name='more subreads than height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              subread_bases=['ATCG', 'ATCG'],
              subread_expanded_cigars=['MMMM', 'MMMM'],
              pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
              ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
          expected_subreads=np.array([
              test_utils.seq_to_array('ATCG'), [1.0, 2.0, 3.0, 4.0],
              [9.0, 10.0, 11.0, 12.0], [deepconsensus_pb2.Subread.REVERSE] * 4,
              [0.0] * 4, [0.1] * 4, [0.2] * 4, [0.3] * 4, [0.4] * 4
          ]),
          max_passes=1,
          expected_num_passes=1,
      ))
  def test_convert_to_tf_example_inference(self, deepconsensus_input,
                                           max_passes, expected_subreads,
                                           expected_num_passes):
    """Check that tensorflow examples are correctly generated."""
    inference = True
    with test_pipeline.TestPipeline() as p:
      example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
      tf_examples = (
          p
          | 'create_data' >> beam.Create([deepconsensus_input])
          | 'convert_to_tf_examples' >> beam.ParDo(
              tf_example_transforms.ConvertToTfExamplesDoFn(
                  example_height=example_height, inference=inference)))

      # Cast expected subreads and labels to correct data type.
      expected_subreads_shape = list(expected_subreads.shape) + [1]
      beam_testing_util.assert_that(
          tf_examples,
          self._tensorflow_example_is_valid(
              deepconsensus_input,
              expected_subreads.astype(dc_constants.NP_DATA_TYPE),
              expected_subreads_shape,
              expected_num_passes,
              None,
              None,
              inference=inference))


class SubreadPermutationsTest(parameterized.TestCase):

  def _permute_check(self, expected_permutations, expected_first_example):

    def _check(output):
      n_permutations = len(output)
      unique_permutations = len(set([str(x.subreads) for x in output]))
      # Check that first example is not shuffled
      output_first_example = [x.bases for x in output[0].subreads]
      self.assertEqual(output_first_example, expected_first_example)
      # Check that all permutations are unique.
      self.assertEqual(n_permutations, unique_permutations)
      # Check that n_permutations is equal to expected
      self.assertEqual(n_permutations, expected_permutations)

    return _check

  @parameterized.named_parameters(
      dict(
          testcase_name='permutations_less_than_passes',
          n_passes=10,
          n_permutations=4,
          expected_permutations=5),
      dict(
          testcase_name='permutations_more_than_passes',
          n_passes=3,
          n_permutations=20,
          expected_permutations=6  # Permutations w/o replacement = 3!
      ),
      dict(
          testcase_name='single_subread',
          n_passes=1,
          n_permutations=10,
          expected_permutations=1))
  def test_permute_subreads(self, n_passes, n_permutations,
                            expected_permutations):
    """Tests deepconsensus_pb2.DeepConsensusInput protos created correctly."""
    with test_pipeline.TestPipeline() as p:
      sequence = 'AAAGGGCCCTTT'
      n_bases = len(sequence)
      subreads = []  # Used to track seen sequences
      for bases in itertools.permutations(sequence):
        subread = ''.join(bases)
        if subread not in subreads:
          if len(subreads) >= n_passes:
            break
          subreads.append(subread)

      n_subreads = len(subreads)
      dc_proto = test_utils.make_deepconsensus_input(
          label_bases='A' * n_bases,
          label_expanded_cigar='M' * n_bases,
          subread_bases=subreads,
          subread_expanded_cigars=['M' * n_bases] * len(subreads),
          pws=[[1] * n_bases] * n_subreads,
          ips=[[9] * n_bases] * n_subreads,
          sn=[0.1, 0.2, 0.3, 0.4],
          subread_strand=[deepconsensus_pb2.Subread.REVERSE] * n_subreads)

      output = (
          p
          | beam.Create([dc_proto])
          | beam.ParDo(
              tf_example_transforms.SubreadPermutationsDoFn(
                  n_permutations=n_permutations)))
      beam_testing_util.assert_that(
          output, self._permute_check(expected_permutations, subreads))


class DownSampleTest(absltest.TestCase):

  def test_subsample_inputs(self):
    dc_proto = test_utils.make_deepconsensus_input(
        label_bases='A',
        label_expanded_cigar='M',
        subread_bases=['A'],
        subread_expanded_cigars=['M'],
        pws=[[1]],
        ips=[[9]],
        sn=[0.1, 0.2, 0.3, 0.4])
    n_proto = 400
    downsample_rate = 0.25
    dc_input = [dc_proto] * n_proto
    with test_pipeline.TestPipeline() as p:
      _ = (
          p
          | beam.Create(dc_input)
          | beam.ParDo(
              tf_example_transforms.DownSample(downsample_rate,
                                               'input_dataset')))
    full_count, downsample_count = p.run().metrics().query()['counters']
    self.assertEqual(full_count.result, n_proto)
    # Valid if range falls between 50-150
    self.assertAlmostEqual(downsample_count.result, n_proto * downsample_rate,
                           -2)


if __name__ == '__main__':
  absltest.main()
