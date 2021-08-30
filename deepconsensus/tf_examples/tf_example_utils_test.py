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
"""Tests for deepconsensus.tf_examples.tf_example_utils."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
import numpy as np
import tensorflow as tf

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_transforms
from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils

from nucleus.protos import bed_pb2


class ReverseComplementTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='uppercase bases in vocab',
          sequence='ATCGANN',
          expected_reverse_complement='NNTCGAT'),
      dict(
          testcase_name='uppercase and lowercase bases in vocab',
          sequence='ATCGaat',
          expected_reverse_complement='attCGAT'),
      dict(
          testcase_name='some bases outside vocab',
          sequence='ATCGRRR',
          expected_reverse_complement='RRRCGAT'),
      dict(testcase_name='empty', sequence='', expected_reverse_complement=''),
  )
  def test_reverse_complement(self, sequence, expected_reverse_complement):
    """Checks that we correctly return the reverse complement sequence."""
    reverse_complement = tf_example_utils.reverse_complement(sequence)
    self.assertEqual(reverse_complement, expected_reverse_complement)


class GetRefAndStartAndOffset(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='forward strand',
          forward_ref_sequence='ATCG',
          dc_input=test_utils.make_deepconsensus_input(
              chrom_start=10,
              chrom_end=14,
              strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND),
          expected_outputs=('ATCG', 10, 1)),
      dict(
          testcase_name='reverse strand',
          forward_ref_sequence='ATCG',
          dc_input=test_utils.make_deepconsensus_input(
              chrom_start=10,
              chrom_end=14,
              strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND),
          expected_outputs=('CGAT', 14, -1)),
  )
  def test_get_ref_and_start_and_offset(self, forward_ref_sequence, dc_input,
                                        expected_outputs):
    """Checks that ref sequence orientation, start, and offset are correct."""
    strand = dc_input.strand
    chrom_start = dc_input.chrom_start
    chrom_end = dc_input.chrom_end
    outputs = tf_example_utils.get_ref_and_start_and_offset(
        forward_ref_sequence, strand, chrom_start, chrom_end)
    self.assertEqual(outputs, expected_outputs)

  def test_missing_strand_raises_error(self):
    """Checks that an unspecified strand results in an error."""
    forward_ref_sequence = 'ATCG'
    chrom_start = 10
    chrom_end = 14
    strand = bed_pb2.BedRecord.Strand.NO_STRAND
    with self.assertRaises(ValueError):
      tf_example_utils.get_ref_and_start_and_offset(forward_ref_sequence,
                                                    strand, chrom_start,
                                                    chrom_end)


class GetSequenceWithoutGapsOrPaddingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no padding or gaps',
          label_sequence='ATCGAA',
          expected_output_label='ATCGAA'),
      dict(
          testcase_name='some padding and gaps',
          label_sequence=f'{dc_constants.GAP_OR_PAD}ATCG{dc_constants.GAP_OR_PAD}AA',
          expected_output_label='ATCGAA'),
      dict(testcase_name='empty', label_sequence='', expected_output_label=''),
  )
  def test_get_sequence_without_gaps_or_padding(self, label_sequence,
                                                expected_output_label):
    """Checks that gaps and padding removed from sequence."""
    label = tf_example_utils.get_sequence_without_gaps_or_padding(
        label_sequence)
    self.assertEqual(label, expected_output_label)


class GetLabelStartEndTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='forward strand',
          label_base_positions=[3, 4, 5, 6, 7],
          strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND,
          expected_start_and_end=(3, 8)),
      dict(
          testcase_name='reverse strand',
          label_base_positions=[7, 6, 5, 4, 3],
          strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND,
          expected_start_and_end=(2, 7)),
      dict(
          testcase_name='forward strand with gaps',
          label_base_positions=[-1, 3, 4, -1, 5, 6, 7],
          strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND,
          expected_start_and_end=(3, 8)),
      dict(
          testcase_name='reverse strand with gaps',
          label_base_positions=[-1, 7, 6, 5, 4, -1, 3],
          strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND,
          expected_start_and_end=(2, 7)),
      dict(
          testcase_name='forward strand one value',
          label_base_positions=[-1, -1, 7],
          strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND,
          expected_start_and_end=(7, 8)),
      dict(
          testcase_name='reverse strand one value',
          label_base_positions=[-1, -1, 7],
          strand=bed_pb2.BedRecord.Strand.REVERSE_STRAND,
          expected_start_and_end=(6, 7)),
      dict(
          testcase_name='only gaps',
          label_base_positions=[-1, -1, -1],
          strand=bed_pb2.BedRecord.Strand.FORWARD_STRAND,
          expected_start_and_end=(None, None)),
  )
  def test_get_label_start_end(self, label_base_positions, strand,
                               expected_start_and_end):
    """Checks that correct range returned for label."""
    start_and_end = tf_example_utils.get_label_start_end(
        label_base_positions, strand)
    self.assertEqual(start_and_end, expected_start_and_end)

  def test_missing_strand_raises_error(self):
    """Checks that an unspecified strand results in an error."""
    label_base_positions = [3, 4, 5]
    strand = bed_pb2.BedRecord.Strand.NO_STRAND
    with self.assertRaises(ValueError):
      tf_example_utils.get_label_start_end(label_base_positions, strand)


class MockCounter(object):
  """A mock counter."""

  def __init__(self):
    self.count = 0

  def inc(self):
    self.count += 1


class DeepconsensusInputToExampleTest(parameterized.TestCase):

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
          max_passes=2,
          expected_subreads=np.array([[1.0, 2.0, 3.0,
                                       4.0], [1.0, 2.0, 3.0, 4.0],
                                      [1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0],
                                      [13.0, 14.0, 15.0, 16.0],
                                      [deepconsensus_pb2.Subread.REVERSE] * 4,
                                      [deepconsensus_pb2.Subread.FORWARD] * 4,
                                      [0.0] * 4, [0.1, 0.1, 0.1, 0.1],
                                      [0.2, 0.2, 0.2,
                                       0.2], [0.3, 0.3, 0.3, 0.3],
                                      [0.4, 0.4, 0.4, 0.4]]),
          expected_num_passes=2,
          expected_label=np.array([1.0, 2.0, 3.0, 4.0]),
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
          max_passes=2,
          expected_subreads=np.array([[1.0, 2.0, 3.0, 4.0],
                                      [float(dc_constants.GAP_OR_PAD_INT)] * 4,
                                      [1.0, 2.0, 3.0, 4.0],
                                      [float(dc_constants.GAP_OR_PAD_INT)] * 4,
                                      [5.0, 6.0, 7.0, 8.0],
                                      [float(dc_constants.GAP_OR_PAD_INT)] * 4,
                                      [deepconsensus_pb2.Subread.REVERSE] * 4,
                                      [deepconsensus_pb2.Subread.NO_STRAND] * 4,
                                      [0.0, 0.0, 0.0,
                                       0.0], [0.1, 0.1, 0.1, 0.1],
                                      [0.2, 0.2, 0.2,
                                       0.2], [0.3, 0.3, 0.3, 0.3],
                                      [0.4, 0.4, 0.4, 0.4]]),
          expected_num_passes=1,
          expected_label=np.array([1.0, 2.0, 3.0, 4.0]),
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
              subread_strand=[
                  deepconsensus_pb2.Subread.REVERSE,
                  deepconsensus_pb2.Subread.FORWARD
              ]),
          max_passes=1,
          expected_subreads=np.array([
              [1.0, 2.0, 3.0, 4.0],
              [1.0, 2.0, 3.0, 4.0],
              [9.0, 10.0, 11.0, 12.0],
              [deepconsensus_pb2.Subread.REVERSE] * 4,
              [0.0, 0.0, 0.0, 0.0],
              [0.1, 0.1, 0.1, 0.1],
              [0.2, 0.2, 0.2, 0.2],
              [0.3, 0.3, 0.3, 0.3],
              [0.4, 0.4, 0.4, 0.4],
          ]),
          expected_num_passes=1,
          expected_label=np.array([1.0, 2.0, 3.0, 4.0]),
          expected_label_shape=[4]),
  )
  def test_convert_to_tf_example(self, deepconsensus_input, max_passes,
                                 expected_subreads, expected_num_passes,
                                 expected_label, expected_label_shape):
    """Check that tensorflow examples are correctly generated."""
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    tf_example = tf_example_utils.deepconsensus_input_to_example(
        deepconsensus_input, example_height, inference=False)

    # Cast expected subreads and labels to correct data type.
    subreads_string = tf_example_utils.get_encoded_subreads_from_example(
        tf_example)
    subreads_shape = tf_example_utils.get_subreads_shape_from_example(
        tf_example)
    num_passes = tf_example_utils.get_num_passes_from_example(tf_example)
    label_string = tf_example_utils.get_encoded_label_from_example(tf_example)
    label_shape = tf_example_utils.get_label_shape_from_example(tf_example)
    num_passes = tf_example_utils.get_num_passes_from_example(tf_example)
    dc_input = tf_example_utils.get_encoded_deepconsensus_input_from_example(
        tf_example)
    self.assertEqual(subreads_shape, list(expected_subreads.shape) + [1])
    self.assertEqual(label_shape, expected_label_shape)
    self.assertCountEqual(
        np.ndarray.flatten(expected_subreads),
        tf.io.decode_raw(subreads_string, dc_constants.TF_DATA_TYPE))
    self.assertEqual(num_passes, expected_num_passes)
    self.assertCountEqual(
        expected_label, tf.io.decode_raw(label_string,
                                         dc_constants.TF_DATA_TYPE))
    self.assertEqual(dc_input, deepconsensus_input.SerializeToString())

  @parameterized.named_parameters(
      dict(
          testcase_name='No subreads',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              subread_bases=[],
              subread_expanded_cigars=[],
              pws=[],
              ips=[],
              subread_strand=[],
              sn=[0.1, 0.2, 0.3, 0.4]),
          max_passes=2,
          expected_counters={'examples_no_subreads_counter': 1}),)
  def test_convert_to_tf_example_empty(self, deepconsensus_input, max_passes,
                                       expected_counters):
    """Check that no examples are returned, as expected."""
    counters = {
        'examples_no_subreads_counter': MockCounter(),
        'subreads_outside_vocab_counter': MockCounter(),
        'examples_with_discarded_subreads': MockCounter(),
        'labels_outside_vocab_counter': MockCounter(),
    }
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    tf_example = tf_example_utils.deepconsensus_input_to_example(
        deepconsensus_input=deepconsensus_input,
        example_height=example_height,
        inference=False,
        counters=counters)
    self.assertIsNone(tf_example)
    for k in counters:
      self.assertEqual(counters[k].count, expected_counters.get(k, 0),
                       'key={}'.format(k))

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid example_height',
          deepconsensus_input=test_utils.make_deepconsensus_input(
              label_bases='ATCG',
              label_expanded_cigar='MMMM',
              subread_bases=['ATCG', 'ATCG'],
              subread_expanded_cigars=['MMMM', 'MMMM'],
              pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
              ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
              sn=[0.1, 0.2, 0.3, 0.4],
              subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2),
          example_height=1,
          expected_msg=('example_height - 5 must be non-negative, and '
                        'divisible by four.'),
      ))
  def test_convert_to_tf_example_raise_errors(self, deepconsensus_input,
                                              example_height, expected_msg):
    """Check that errors are raised, as expected."""
    with self.assertRaisesRegex(ValueError, expected_msg):
      _ = tf_example_utils.deepconsensus_input_to_example(
          deepconsensus_input, example_height, inference=False)


class MetricsToJsonTest(absltest.TestCase):

  def test_metrics_to_json(self):
    """Tests that metric counters are extracted correctly."""

    input_copies = 10
    deepconsensus_input = test_utils.make_deepconsensus_input(
        label_bases='ATCG',
        label_expanded_cigar='MMMM',
        subread_bases=['ATCG', 'ATCG'],
        subread_expanded_cigars=['MMMM', 'MMMM'],
        pws=[[1, 2, 3, 4], [5, 6, 7, 8]],
        ips=[[9, 10, 11, 12], [13, 14, 15, 16]],
        sn=[0.1, 0.2, 0.3, 0.4],
        subread_strand=[deepconsensus_pb2.Subread.REVERSE] * 2)

    max_passes = 1
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    with test_pipeline.TestPipeline() as p:
      _ = (
          p
          | 'create_data' >> beam.Create([deepconsensus_input] * input_copies)
          | 'convert_to_tf_examples' >> beam.ParDo(
              tf_example_transforms.ConvertToTfExamplesDoFn(
                  example_height=example_height, inference=False)))

    path = self.create_tempfile().full_path
    tf_example_utils.metrics_to_json(p.run(), path)
    result = json.load(tf.io.gfile.GFile(path, 'r'))
    self.assertEqual(result['convert_to_tf_examples:total_examples'],
                     input_copies)


if __name__ == '__main__':
  absltest.main()
