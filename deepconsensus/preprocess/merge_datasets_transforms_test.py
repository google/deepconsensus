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
"""Tests for deepconsensus.merge_datasets_transforms."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as beam_testing_util

from deepconsensus.preprocess import merge_datasets_transforms
from nucleus.testing import test_utils
from nucleus.util import struct_utils


class RemoveReadsMissingSequenceDoFnTest(absltest.TestCase):
  """Tests for merge_datasets_transforms.RemoveReadsMissingSequenceDoFn."""

  def test_remove_reads_missing_sequence(self):
    """Tests that only reads with a sequence are kept."""

    with test_pipeline.TestPipeline() as p:
      reads = [
          test_utils.make_read(bases='', start=0, name='read_missing_sequence'),
          test_utils.make_read(
              bases='ATCG', start=0, name='read_with_sequence'),
      ]
      filtered_reads = (
          p
          | beam.Create(reads)
          | beam.ParDo(
              merge_datasets_transforms.RemoveReadsMissingSequenceDoFn()))

      expected = [(reads[1])]
      beam_testing_util.assert_that(filtered_reads,
                                    beam_testing_util.equal_to(expected))


class RemoveIncorrectlyMappedReadsDoFnTest(absltest.TestCase):
  """Tests for merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn."""

  def test_remove_incorrectly_mapped_reads(self):
    """Tests that only reads with same reference and fragment molecules kept."""

    with test_pipeline.TestPipeline() as p:
      reads = [
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/1/ccs',
              name='m54316_180808_005743/1/truth'),
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/3/ccs',
              name='m54316_180808_005743/4/truth'),
      ]
      filtered_reads = (
          p
          | beam.Create(reads)
          | beam.ParDo(
              merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn()))

      expected = [(reads[0])]
      beam_testing_util.assert_that(filtered_reads,
                                    beam_testing_util.equal_to(expected))


class GetReadNameDoFnTest(absltest.TestCase):
  """Tests for merge_datasets_transforms.GetReadNameDoFn."""

  def test_get_read_name(self):
    """Tests that read name is correctly read."""

    with test_pipeline.TestPipeline() as p:
      reads = [
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/1/ccs',
              name='m54316_180808_005743/2/truth'),
          test_utils.make_read(
              bases='ATCG',
              start=0,
              chrom='m54316_180808_005743/3/ccs',
              name='m54316_180808_005743/4/truth'),
      ]
      read_names = (
          p
          | beam.Create(reads)
          | beam.ParDo(merge_datasets_transforms.GetReadNameDoFn()))

      expected = [
          ('m54316_180808_005743/2/truth', reads[0]),
          ('m54316_180808_005743/4/truth', reads[1]),
      ]
      beam_testing_util.assert_that(read_names,
                                    beam_testing_util.equal_to(expected))


class MergeSubreadsDoFnTest(absltest.TestCase):
  """Tests for merge_datasets_transforms.MergeSubreadsDoFn."""

  def test_merge_without_reverse_complementing(self):
    """Tests adding sequence that already exists."""

    with test_pipeline.TestPipeline() as p:
      aligned_reads = [
          test_utils.make_read(name='some read', bases='ATCG', start=0)
      ]

      unaligned_reads = [
          test_utils.make_read(name='some read', bases='ATCG', start=0)
      ]
      struct_utils.set_int_field(unaligned_reads[0].info, 'pw', [0, 1, 2, 3])
      struct_utils.set_int_field(unaligned_reads[0].info, 'ip', [4, 5, 6, 7])
      struct_utils.set_number_field(unaligned_reads[0].info, 'sn',
                                    [0.1, 0.2, 0.3, 0.4])

      merged_reads = (
          p
          | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
          | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(unaligned_reads))

  def test_merge_with_reverse_complementing(self):
    """Tests adding sequence that already exists with reverse complementing."""

    with test_pipeline.TestPipeline() as p:
      aligned_reads = [
          test_utils.make_read(name='some read', bases='GTCA', start=0)
      ]

      unaligned_reads = [
          test_utils.make_read(name='some read', bases='TGAC', start=3)
      ]
      struct_utils.set_int_field(unaligned_reads[0].info, 'pw', [0, 1, 2, 3])
      struct_utils.set_int_field(unaligned_reads[0].info, 'ip', [4, 5, 6, 7])
      struct_utils.set_number_field(unaligned_reads[0].info, 'sn',
                                    [0.1, 0.2, 0.3, 0.4])

      merged_reads = (
          p
          | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
          | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))

      expected_reads = [
          test_utils.make_read(name='some read', bases='GTCA', start=0)
      ]
      struct_utils.set_int_field(expected_reads[0].info, 'pw', [3, 2, 1, 0])
      struct_utils.set_int_field(expected_reads[0].info, 'ip', [7, 6, 5, 4])
      struct_utils.set_number_field(expected_reads[0].info, 'sn',
                                    [0.1, 0.2, 0.3, 0.4])
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(expected_reads))

  def test_multiple_aligned_reads(self):
    """Tests that first alignment used when multiple read alignments exist."""
    with test_pipeline.TestPipeline() as p:
      aligned_reads = [
          test_utils.make_read(name='some read', bases='ATCG', start=0),
          test_utils.make_read(name='some read', bases='GGGG', start=0)
      ]

      unaligned_reads = [
          test_utils.make_read(name='some read', bases='ATCG', start=0)
      ]

      # These fields will appear in the output, but will be empty as the input
      # does not contain these values.
      struct_utils.set_int_field(unaligned_reads[0].info, 'pw', [])
      struct_utils.set_int_field(unaligned_reads[0].info, 'ip', [])
      struct_utils.set_number_field(unaligned_reads[0].info, 'sn', [])
      merged_reads = (
          p
          | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
          | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(unaligned_reads))

  def test_reads_with_incompatible_sequences_skipped(self):
    """Tests that reads with incompatible sequences are not merged."""

    with test_pipeline.TestPipeline() as p:

      aligned_reads = [
          test_utils.make_read(name='some read', bases='AAAA', start=0)
      ]

      unaligned_reads = [
          test_utils.make_read(name='some read', bases='ATCG', start=3)
      ]

      merged_reads = (
          p
          | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
          | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to([]))

  def test_multiple_unaligned_reads_throws_error(self):
    """Tests that error thrown when multiple unaligned reads exist."""

    with self.assertRaises(ValueError):
      with test_pipeline.TestPipeline() as p:
        aligned_reads = [
            test_utils.make_read(name='some read', bases='ATCG', start=0)
        ]

        unaligned_reads = [
            test_utils.make_read(name='some read', bases='ATCG', start=0),
            test_utils.make_read(name='some read', bases='ATCG', start=0)
        ]

        _ = (
            p
            | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
            | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))

  def test_missing_aligned_sequence_throws_error(self):
    """Tests that merging aligned read missing sequence throws an error."""

    with self.assertRaises(ValueError):
      with test_pipeline.TestPipeline() as p:

        aligned_reads = [
            test_utils.make_read(name='some read', bases='', start=0)
        ]

        unaligned_reads = [
            test_utils.make_read(name='some read', bases='ATCG', start=3)
        ]

        _ = (
            p
            | beam.Create([('some molecule', (aligned_reads, unaligned_reads))])
            | beam.ParDo(merge_datasets_transforms.MergeSubreadsDoFn()))


class MergeLabelsDoFnTest(absltest.TestCase):

  def test_add_not_missing_sequence(self):
    """Tests adding sequence into label which already contains a sequence."""

    with test_pipeline.TestPipeline() as p:
      reads = [test_utils.make_read(name='some read', bases='ATCG', start=0)]
      sequences = ['ATCG']
      merged_reads = (
          p
          | beam.Create([('some molecule', (reads, sequences))])
          | beam.ParDo(merge_datasets_transforms.MergeLabelsDoFn()))
      expected = [reads[0]]
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(expected))

  def test_add_missing_sequence(self):
    """Tests adding sequence into label which does not contain a sequence."""

    with test_pipeline.TestPipeline() as p:
      reads = [test_utils.make_read(name='some read', bases='', start=0)]
      sequences = ['ATCG']
      merged_reads = (
          p
          | beam.Create([('some molecule', (reads, sequences))])
          | beam.ParDo(merge_datasets_transforms.MergeLabelsDoFn()))
      expected = [test_utils.make_read(name='some read', bases='ATCG', start=0)]
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(expected))

  def test_multiple_aligned_labels(self):
    """Tests that first alignment used when multiple label alignments exist."""

    with test_pipeline.TestPipeline() as p:
      reads = [
          test_utils.make_read(name='some read', bases='', start=0),
          test_utils.make_read(name='some read', bases='GGGG', start=0),
      ]
      sequences = ['ATCG']
      merged_reads = (
          p
          | beam.Create([('some molecule', (reads, sequences))])
          | beam.ParDo(merge_datasets_transforms.MergeLabelsDoFn()))
      expected = [test_utils.make_read(name='some read', bases='ATCG', start=0)]
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to(expected))

  def test_multiple_sequences_throws_error(self):
    """Tests that multiple sequence for a given label throws an error."""

    with self.assertRaises(ValueError):
      with test_pipeline.TestPipeline() as p:
        reads = [test_utils.make_read(name='some read', bases='ATCG', start=0)]
        sequences = ['ATCG', 'ATCG']
        _ = (
            p
            | beam.Create([('some molecule', (reads, sequences))])
            | beam.ParDo(merge_datasets_transforms.MergeLabelsDoFn()))

  def test_different_sequence_is_discarded(self):
    """Tests that label with different sequence is discarded."""

    with test_pipeline.TestPipeline() as p:
      reads = [test_utils.make_read(name='some read', bases='ATAT', start=0)]
      sequences = ['ATCG']
      merged_reads = (
          p
          | beam.Create([('some molecule', (reads, sequences))])
          | beam.ParDo(merge_datasets_transforms.MergeLabelsDoFn()))
      beam_testing_util.assert_that(merged_reads,
                                    beam_testing_util.equal_to([]))


if __name__ == '__main__':
  absltest.main()
