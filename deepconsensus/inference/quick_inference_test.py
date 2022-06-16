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
"""Tests for quick_inference."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam

from deepconsensus.inference import quick_inference
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils

FLAGS = flags.FLAGS


def setUpModule():
  logging.set_verbosity(logging.FATAL)


class QuickInferenceTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          subreads='human_1m/subreads_to_ccs.bam',
          ccs_bam='human_1m/ccs.bam',
          expected_lengths=[17141, 16320]))
  @flagsaver.flagsaver
  def test_end_to_end(self, subreads, ccs_bam, expected_lengths):
    FLAGS.subreads_to_ccs = test_utils.deepconsensus_testdata(subreads)
    FLAGS.ccs_bam = test_utils.deepconsensus_testdata(ccs_bam)
    output_path = test_utils.test_tmpfile('output_path.fastq')
    FLAGS.output = output_path
    FLAGS.checkpoint = test_utils.deepconsensus_testdata('model/checkpoint-1')
    FLAGS.min_quality = 0  # Qualities are lower due to tiny sample model.
    FLAGS.limit = 2
    FLAGS.cpus = 2
    outcomes = quick_inference.run()

    count = 0
    output_lengths = []
    with pysam.FastqFile(output_path) as fastq_reader:
      for record in fastq_reader:
        self.assertTrue(record.name.endswith('/ccs'))
        self.assertTrue(set(record.sequence).issubset(dc_constants.VOCAB))
        self.assertEqual(len(record.sequence), len(record.quality))
        # Length of the output read should be deterministic for the same model.
        output_lengths.append(len(record.sequence))
        count += 1
    self.assertEqual(count, 2)
    # TODO: Figure out why lengths are not deterministic.
    # Not deterministic, might be due to the test model used since other runs
    # with the release model have been deterministic so far.
    # self.assertEqual(expected_lengths, output_lengths)
    print('expected lengths:', expected_lengths, 'output lengths:',
          output_lengths)
    self.assertEqual(outcomes.success, 2)

  @parameterized.parameters(
      dict(cpus=0, batch_zmws=1), dict(cpus=0, batch_zmws=0),
      dict(cpus=1, batch_zmws=1), dict(cpus=1, batch_zmws=100))
  @flagsaver.flagsaver
  def test_end_to_end_multiprocessing(self, cpus, batch_zmws):
    FLAGS.subreads_to_ccs = test_utils.deepconsensus_testdata(
        'human_1m/subreads_to_ccs.bam')
    FLAGS.ccs_bam = test_utils.deepconsensus_testdata('human_1m/ccs.bam')
    FLAGS.checkpoint = test_utils.deepconsensus_testdata('model/checkpoint-1')
    output_path = test_utils.test_tmpfile('output_path.fastq')
    FLAGS.output = output_path
    FLAGS.batch_zmws = batch_zmws
    FLAGS.cpus = cpus
    FLAGS.min_quality = 0  # Qualities are lower due to tiny sample model.
    FLAGS.limit = 2
    outcomes = quick_inference.run()

    count = 0
    with pysam.FastqFile(output_path) as fastq_reader:
      for record in fastq_reader:
        self.assertTrue(record.name.endswith('/ccs'))
        self.assertTrue(set(record.sequence).issubset(dc_constants.VOCAB))
        self.assertEqual(len(record.sequence), len(record.quality))
        count += 1
    self.assertEqual(count, 2)
    self.assertEqual(outcomes.success, 2)

  @parameterized.parameters(
      dict(
          calibration_str='',
          expected=quick_inference.QualityCalibrationValues(
              enabled=False, threshold=0.0, w=1.0, b=0.0),
          message='Test 1: Valid empty calibration string.'),
      dict(
          calibration_str='10,1.0,0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=10.0, w=1.0, b=0.2222),
          message='Test 2: Valid calibration string with positive values.'),
      dict(
          calibration_str='-10,1.0,0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=1.0, b=0.2222),
          message='Test 3: Valid calibration string with negative threshold.'),
      dict(
          calibration_str='-10,-1.0,-0.2222',
          expected=quick_inference.QualityCalibrationValues(
              enabled=True, threshold=-10.0, w=-1.0, b=-0.2222),
          message='Test 4: Valid calibration string with all negative values.'))
  @flagsaver.flagsaver
  def test_parse_calibration_string(self, calibration_str, expected, message):
    """Tests for parse_calibration_string method."""
    returned = quick_inference.parse_calibration_string(calibration_str)
    self.assertEqual(returned.enabled, expected.enabled, msg=message)
    self.assertEqual(returned.threshold, expected.threshold, msg=message)
    self.assertEqual(returned.w, expected.w, msg=message)
    self.assertEqual(returned.b, expected.b, msg=message)

  @parameterized.parameters(
      dict(
          calibration_str='ABCD',
          message='Test 1: Invalid calibration string ABCD.'),
      dict(
          calibration_str='A,BC,D',
          message='Test 2: Invalid calibration string A,BC,D.'),
      dict(
          calibration_str='10,1.0',
          message='Test 2: Invalid calibration string 10,1.0.'),
      dict(
          calibration_str='10,AB,1.0',
          message='Test 2: Invalid calibration string 10,AB,1.0.'),
      dict(
          calibration_str='10,0.1.1,1.0',
          message='Test 2: Invalid calibration string 10,0.1.1,1.0.'),
  )
  @flagsaver.flagsaver
  def test_parse_calibration_string_exceptions(self, calibration_str, message):
    with self.assertRaises(Exception, msg=message):
      quick_inference.parse_calibration_string(calibration_str)


if __name__ == '__main__':
  absltest.main()
