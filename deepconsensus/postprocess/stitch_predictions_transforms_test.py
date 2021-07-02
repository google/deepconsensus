"""Tests for deepconsensus .postprocess.stitch_predictions_transforms."""

import copy
import random

from absl.testing import absltest
from absl.testing import parameterized

import apache_beam as beam
from apache_beam.testing import test_pipeline
import apache_beam.testing.util as beam_testing_util

from deepconsensus.postprocess import stitch_predictions_transforms
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils
from deepconsensus.utils import utils


class GetFullSequenceDoFnTest(parameterized.TestCase):

  def test_get_full_sequences(self):
    """Checks that tf.Examples are correctly read in and parsed."""
    molecule_name = 'm54238_180901_011437/7209150'
    dc_inputs = [
        test_utils.make_deepconsensus_input(molecule_start=start)
        for start in range(0, 50, 5)
    ]
    # Use the label as the prediction for this test.
    for dc_input in dc_inputs:
      dc_input.deepconsensus_prediction = dc_input.label.bases
      dc_input.quality_string = '!' * len(dc_input.deepconsensus_prediction)
    full_sequence = ''.join(
        [dc_input.deepconsensus_prediction for dc_input in dc_inputs])
    full_quality_string = ''.join(
        [dc_input.quality_string for dc_input in dc_inputs])
    dc_inputs_shuffled = copy.deepcopy(dc_inputs)
    random.shuffle(dc_inputs_shuffled)
    molecule_name_and_deepconsensus_inputs = (molecule_name, dc_inputs_shuffled)
    expected_output = (molecule_name, full_sequence, full_quality_string)
    example_width = len(dc_inputs_shuffled[0].label.bases)
    with test_pipeline.TestPipeline() as p:
      output = (
          p
          | beam.Create([molecule_name_and_deepconsensus_inputs])
          | beam.ParDo(
              stitch_predictions_transforms.GetFullSequenceDoFn(
                  example_width=example_width)))
      beam_testing_util.assert_that(
          output, beam_testing_util.equal_to([expected_output]))

  def test_get_partial_sequences(self):
    """Read and parse tf.Examples with some missing seq windows."""
    molecule_name = 'm54238_180901_011437/7209150'
    dc_inputs = [
        test_utils.make_deepconsensus_input(molecule_start=start)
        for start in range(0, 50, 5)
    ]
    # Select a random sequence to knockout
    # This does not work for the final window.
    rand_seq_knockout = random.randint(0, len(dc_inputs) - 2)
    # Use the label as the prediction for this test.
    for n, dc_input in enumerate(dc_inputs):
      if n == rand_seq_knockout:
        dc_input.deepconsensus_prediction = 'N' * len(dc_input.label.bases)
        dc_input.quality_string = utils.quality_score_to_string(
            dc_constants.EMPTY_QUAL) * len(dc_input.deepconsensus_prediction)
      else:
        dc_input.deepconsensus_prediction = dc_input.label.bases
        dc_input.quality_string = '!' * len(dc_input.deepconsensus_prediction)
    full_sequence = ''.join(
        [dc_input.deepconsensus_prediction for dc_input in dc_inputs])
    full_quality_string = ''.join(
        [dc_input.quality_string for dc_input in dc_inputs])
    # Knockout sequence
    dc_inputs.pop(rand_seq_knockout)
    dc_inputs_shuffled = copy.deepcopy(dc_inputs)
    random.shuffle(dc_inputs_shuffled)
    molecule_name_and_deepconsensus_inputs = (molecule_name, dc_inputs_shuffled)
    expected_output = (molecule_name, full_sequence, full_quality_string)
    example_width = len(dc_inputs_shuffled[0].label.bases)
    with test_pipeline.TestPipeline() as p:
      output = (
          p
          | beam.Create([molecule_name_and_deepconsensus_inputs])
          | beam.ParDo(
              stitch_predictions_transforms.GetFullSequenceDoFn(
                  example_width=example_width, fill_n=True)))
      beam_testing_util.assert_that(
          output, beam_testing_util.equal_to([expected_output]))

  @parameterized.parameters([False, True])
  def test_missing_windows_behavior_diff_widths(self, correct_width):
    """This test case illustrates a failure mode when the width is not correct.

    We discard most of our dc_inputs in the input data, so the molecule should
    be thrown out, since not all windows are present. However, when the width is
    not set correctly, the molecule is not correctly thrown out.

    Args:
      correct_width: if True, the correct width is used (not including padding).
        If False, the wrong width is used (length of bases and padding).
    """
    molecule_name = 'm54238_180901_011437/7209150'
    dc_inputs = [
        test_utils.make_deepconsensus_input(molecule_start=start)
        for start in range(0, 50, 5)
    ]
    # Keep two non-adjacent windows of the full molecule.
    dc_inputs = dc_inputs[::2]
    # Use the label as the prediction for this test. Add some padding too.
    for dc_input in dc_inputs:
      dc_input.deepconsensus_prediction = dc_input.label.bases + '-' * 5
    if correct_width:
      example_width = len(dc_inputs[0].label.bases)
    else:
      # Note: example width being set incorrectly won't always result in the
      # example being kept incorrectly. Sometimes, even with an incorrect
      # width, we do still filter the example out correctly. For this test case,
      # the chosen width will incorrectly result in the example being kept.
      example_width = len(dc_inputs[0].deepconsensus_prediction)
    with test_pipeline.TestPipeline() as p:
      output = (
          p
          | beam.Create([(molecule_name, dc_inputs)])
          | beam.ParDo(
              stitch_predictions_transforms.GetFullSequenceDoFn(
                  example_width=example_width)))
      if correct_width:
        beam_testing_util.assert_that(output, beam_testing_util.is_empty())
      else:
        beam_testing_util.assert_that(output, beam_testing_util.is_not_empty())


class ConvertToFastqStrDoFnTest(absltest.TestCase):

  def _is_valid_fastq_str(self, molecule_name):

    def _check_fasta_str(outputs):
      fasta_str = outputs[0]
      fasta_str_parts = fasta_str.split('\n')
      contig_name = fasta_str_parts[0]
      # Check that the contig name is formatted correctly.
      self.assertEqual(contig_name, '@' + molecule_name + '/ccs')
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

    return _check_fasta_str

  def test_convert_to_fastq_str(self):
    """Checks that the FASTQ is correctly formatted."""
    molecule_name = 'm54238_180901_011437/7209150'
    length = 10000
    sequence = 'ATCG' * length
    quality_string = utils.quality_score_to_string(10) * 4 * length
    input_data = (molecule_name, sequence, quality_string)
    with test_pipeline.TestPipeline() as p:
      output = (
          p
          | beam.Create([
              input_data,
          ])
          | beam.ParDo(stitch_predictions_transforms.ConvertToFastqStrDoFn()))
      beam_testing_util.assert_that(output,
                                    self._is_valid_fastq_str(molecule_name))


if __name__ == '__main__':
  absltest.main()
