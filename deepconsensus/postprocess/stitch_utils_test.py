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
"""Tests for deepconsensus.postprocess.stitch_utils."""

import copy
import random

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from deepconsensus.postprocess import stitch_utils
from deepconsensus.utils import dc_constants
from deepconsensus.utils import utils


def setUpModule():
  logging.set_verbosity(logging.FATAL)


def fake_model_output(start: int, window_size: int, padding: int):
  return stitch_utils.DCModelOutput(
      molecule_name='name',
      window_pos=start,
      sequence=''.join(random.choices('ACGT', k=window_size)) + '-' * padding,
      quality_string='!' * (window_size + padding))


def fake_model_outputs(window_size: int, num_windows: int, padding: int = 0):
  return [
      fake_model_output(start=start, window_size=window_size, padding=padding)
      for start in range(0, window_size * num_windows, window_size)
  ]


class GetFullSequenceTest(parameterized.TestCase):

  def test_get_full_sequences(self):
    """Checks that tf.Examples are correctly read in and parsed."""
    width = 5
    dc_outputs = fake_model_outputs(window_size=width, num_windows=10)
    expected_sequence = ''.join(
        [dc_output.sequence for dc_output in dc_outputs])
    expected_quality_string = ''.join(
        [dc_output.quality_string for dc_output in dc_outputs])
    dc_outputs_shuffled = copy.deepcopy(dc_outputs)
    random.shuffle(dc_outputs_shuffled)

    sequence_output, quality_output = stitch_utils.get_full_sequence(
        deepconsensus_outputs=dc_outputs_shuffled, example_width=width)
    self.assertEqual(expected_sequence, sequence_output)
    self.assertEqual(expected_quality_string, quality_output)

  def test_get_partial_sequences(self):
    """Read and parse tf.Examples with some missing seq windows."""
    width = 5
    dc_outputs = fake_model_outputs(window_size=width, num_windows=10)
    # Select a random sequence to knockout
    # This does not work for the final window.
    rand_seq_knockout = random.randint(0, len(dc_outputs) - 2)
    for n, dc_output in enumerate(dc_outputs):
      if n == rand_seq_knockout:
        dc_output.sequence = 'N' * width
        dc_output.quality_string = utils.quality_score_to_string(
            dc_constants.EMPTY_QUAL) * len(dc_output.sequence)
    expected_sequence = ''.join(
        [dc_output.sequence for dc_output in dc_outputs])
    expected_quality_string = ''.join(
        [dc_output.quality_string for dc_output in dc_outputs])
    # Knockout sequence
    dc_outputs.pop(rand_seq_knockout)
    dc_outputs_shuffled = copy.deepcopy(dc_outputs)
    random.shuffle(dc_outputs_shuffled)

    sequence_output, quality_output = stitch_utils.get_full_sequence(
        deepconsensus_outputs=dc_outputs_shuffled,
        example_width=width,
        fill_n=True)
    self.assertEqual(expected_sequence, sequence_output)
    self.assertEqual(expected_quality_string, quality_output)


class RemoveGapsAndPaddingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no gaps/padding',
          sequence='ATCG',
          quality_string=utils.quality_scores_to_string([1, 2, 3, 4]),
          expected_sequence='ATCG',
          expected_quality_string=utils.quality_scores_to_string([1, 2, 3, 4]),
      ),
      dict(
          testcase_name='some gaps/padding',
          sequence='AT CG ',
          quality_string=utils.quality_scores_to_string([1, 2, 3, 4, 5, 6]),
          expected_sequence='ATCG',
          expected_quality_string=utils.quality_scores_to_string([1, 2, 4, 5]),
      ),
      dict(
          testcase_name='all gaps/padding',
          sequence='    ',
          quality_string=utils.quality_scores_to_string([1, 2, 3, 4]),
          expected_sequence=None,
          expected_quality_string=None,
      ),
  )
  def test_remove_gaps_and_padding(self, sequence, quality_string,
                                   expected_sequence, expected_quality_string):
    if expected_sequence:
      expected_output = (expected_sequence, expected_quality_string)
    else:
      expected_output = ('', '')

    output = stitch_utils.remove_gaps_and_padding(
        sequence=sequence, quality_string=quality_string)
    self.assertEqual(output, expected_output)


class IsQualityAboveThresholdTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(min_quality=20, read_qualities=(19, 19, 19, 19), should_pass=False),
      dict(min_quality=20, read_qualities=(20, 20, 20, 20), should_pass=True),
      dict(min_quality=40, read_qualities=(40, 40, 40, 40), should_pass=True),
      # Average phred is not the same as average base quality.
      dict(min_quality=40, read_qualities=(39, 39, 41, 41), should_pass=False)
  ])
  def test_is_quality_above_threshold(self, min_quality, read_qualities,
                                      should_pass):
    quality_string = [utils.quality_score_to_string(x) for x in read_qualities]

    passes = stitch_utils.is_quality_above_threshold(
        quality_string=quality_string, min_quality=min_quality)
    self.assertEqual(passes, should_pass)


class ConvertToFastqStrDoFnTest(absltest.TestCase):

  def assert_correct_fastq_str(self, molecule_name, fasta_str):
    fasta_str_parts = fasta_str.split('\n')
    contig_name = fasta_str_parts[0]
    # Check that the contig name is formatted correctly.
    self.assertEqual(contig_name, '@' + molecule_name)
    sequence_line = fasta_str_parts[1]
    separator_line = fasta_str_parts[2]
    quality_string_line = fasta_str_parts[3]
    padding_and_gap = [dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD]
    # Check the sequence line contains only valid bases.
    self.assertNoCommonElements(set(sequence_line), padding_and_gap)
    self.assertEqual(separator_line, '+')
    # Not all values in this range are allowed, since we are binning, but we
    # do not consider that for this test.
    possible_quals = utils.quality_scores_to_string(
        list(range(dc_constants.EMPTY_QUAL, dc_constants.MAX_QUAL + 1)))
    self.assertContainsSubset(quality_string_line, possible_quals)
    self.assertLen(quality_string_line, len(sequence_line))

  def test_convert_to_fastq_str(self):
    """Checks that the FASTQ is correctly formatted."""
    molecule_name = 'm54238_180901_011437/7209150/ccs'
    length = 10000
    sequence = 'ATCG' * length
    quality_string = utils.quality_score_to_string(10) * 4 * length
    output = stitch_utils.format_as_fastq(
        molecule_name=molecule_name,
        sequence=sequence,
        quality_string=quality_string)

    self.assert_correct_fastq_str(molecule_name=molecule_name, fasta_str=output)


if __name__ == '__main__':
  absltest.main()
